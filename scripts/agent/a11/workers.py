from __future__ import annotations

import atexit
import contextlib
import ctypes
import math
import multiprocessing as mp
import os
import random
import signal
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from queue import Empty, Full
from typing import Any

from .config import (
    BACKEND_API_VERSION,
    BACKEND_BUILD_DIR,
    BACKEND_DIR,
    ENV_COUNT,
    ENVS_PER_FILE,
    ENUM_TIME_BUDGET_S,
    ENV_MAX_CONSECUTIVE_FAILURES,
    ENV_RESTART_TIMEOUT_S,
    GPU_IDS,
    PROJECT_ROOT,
    SEED,
    SIEVE_B1_REL_IMPROVEMENT,
    SIEVE_FREE_DIM,
    SIEVE_FREE_DIM_CAP,
    SIEVE_KEEP_WORKDIR,
    SIEVE_LOGPOT_IMPROVEMENT,
    SIEVE_MAX_CANDIDATES,
    SIEVE_MAX_PAIRS,
    SIEVE_MAX_ROUNDS,
    SIEVE_MEMORY_BUDGET_MB,
    SIEVE_QUEUE_WAIT_TIMEOUT_S,
    SIEVE_QUEUE_SIZE,
    SIEVE_RESPONSE_POLL_SECONDS,
    SIEVE_REQUEST_PUT_TIMEOUT_S,
    SIEVE_RESPONSE_PUT_TIMEOUT_S,
    SIEVE_RESPONSE_TIMEOUT_S,
    SIEVE_SERVICE_CLOSE_SECONDS,
    SIEVE_TIME_BUDGET_S,
    SIEVE_WORKDIR,
    SIEVE_WORKER_MAX_TASKS,
    WORKER_CLOSE_GRACE_SECONDS,
    WORKER_KILL_GRACE_SECONDS,
    WORKER_TERMINATE_GRACE_SECONDS,
)
from .io_utils import parse_dim_seed
from .runtime import configure_env_runtime
from .scheduler import create_cpu_gate


@contextlib.contextmanager
def _silence_native_output():
    """Silence stdout/stderr emitted inside one native sieve invocation.

    The persistent sieve worker is single-threaded, so temporarily redirecting
    process file descriptors 1/2 is safe here and also catches C/C++ printf,
    iostream and child-process output from the vendored sieve engine.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "wb", buffering=0) as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            try:
                yield
            finally:
                # Flush libc/C++ stdio while it still points at /dev/null so
                # buffered native progress lines cannot leak after restoration.
                try:
                    ctypes.CDLL(None).fflush(None)
                except Exception:
                    pass
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _format_sieve_shortening(result: dict[str, Any]) -> str:
    try:
        value = float(result.get("b1_relative_improvement", float("nan")))
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:+.6f}%"


# ============================================================
# Environment-side sieve client
# ============================================================


class SieveClient:
    """Synchronous IPC client used by one environment process."""

    def __init__(
        self,
        request_queue,
        response_queue,
        env_id: int,
        gpu_id: int,
        generation: int = 0,
    ):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.env_id = int(env_id)
        self.gpu_id = int(gpu_id)
        self.generation = int(generation)
        self._counter = 0

    def _next_task_id(self) -> int:
        self._counter += 1

        # Task ids must never be reused after an environment respawn: a delayed
        # GPU response from the dead generation may still be in flight.
        # Layout: 8 bits env id | 16 bits generation | 40 bits sequence.
        if self._counter >= (1 << 40):
            raise RuntimeError("sieve task counter exhausted")
        return (self.env_id << 56) | ((self.generation & 0xFFFF) << 40) | self._counter

    def reduce(
        self,
        block_matrix: str,
        beta: int,
    ) -> dict[str, Any]:
        task_id = self._next_task_id()

        request = {
            "cmd": "sieve",
            "task_id": task_id,
            "env_id": self.env_id,
            "gpu_id": self.gpu_id,
            "beta": int(beta),
            "block_matrix": str(block_matrix),
            "budget": {
                "max_candidates": (SIEVE_MAX_CANDIDATES),
                "max_rounds": (SIEVE_MAX_ROUNDS),
                "max_pairs": (SIEVE_MAX_PAIRS),
                "time_budget_s": (SIEVE_TIME_BUDGET_S),
                "memory_budget_mb": (SIEVE_MEMORY_BUDGET_MB),
                "min_b1_rel_improvement": (SIEVE_B1_REL_IMPROVEMENT),
                "min_logpot_improvement": (SIEVE_LOGPOT_IMPROVEMENT),
                "free_dim": (SIEVE_FREE_DIM),
                "free_dim_cap": (SIEVE_FREE_DIM_CAP),
            },
        }

        try:
            self.request_queue.put(
                request,
                timeout=max(0.1, SIEVE_REQUEST_PUT_TIMEOUT_S),
            )
        except Full as exc:
            raise RuntimeError(
                f"GPU{self.gpu_id} sieve request queue remained full for "
                f"{SIEVE_REQUEST_PUT_TIMEOUT_S:.1f}s"
            ) from exc

        deadline = time.monotonic() + max(1.0, SIEVE_QUEUE_WAIT_TIMEOUT_S)
        execution_started = False

        while True:
            try:
                response = self.response_queue.get(
                    timeout=max(
                        0.1,
                        SIEVE_RESPONSE_POLL_SECONDS,
                    )
                )
            except Empty:
                if time.monotonic() >= deadline:
                    phase = "execution" if execution_started else "queue wait"
                    limit = (
                        SIEVE_RESPONSE_TIMEOUT_S
                        if execution_started
                        else SIEVE_QUEUE_WAIT_TIMEOUT_S
                    )
                    raise TimeoutError(
                        f"GPU{self.gpu_id} persistent sieve {phase} timed out "
                        f"after {limit:.1f}s "
                        f"(env_id={self.env_id}, task_id={task_id}, beta={beta})"
                    )
                continue

            received_task_id = int(
                response.get(
                    "task_id",
                    -1,
                )
            )

            if received_task_id < task_id:
                # A delayed response from an already-aborted request must not
                # poison the next request on this environment queue.
                continue

            if received_task_id != task_id:
                raise RuntimeError(
                    "Persistent sieve response mismatch:\n"
                    f"  env_id   = {self.env_id}\n"
                    f"  gpu_id   = {self.gpu_id}\n"
                    f"  expected = {task_id}\n"
                    f"  received = {received_task_id}"
                )

            if response.get("status") == "started":
                execution_started = True
                deadline = time.monotonic() + max(
                    1.0,
                    SIEVE_RESPONSE_TIMEOUT_S,
                )
                continue

            if not bool(
                response.get(
                    "ok",
                    False,
                )
            ):
                error = response.get(
                    "error",
                    "unknown persistent sieve error",
                )

                native_traceback = response.get(
                    "traceback",
                    "",
                )

                raise RuntimeError(
                    f"GPU{self.gpu_id} persistent sieve worker failed:\n"
                    f"{error}\n"
                    f"{native_traceback}"
                )

            result = dict(response["result"])

            result.update(
                {
                    "physical_gpu": self.gpu_id,
                    "sieve_worker_pid": int(
                        response.get(
                            "worker_pid",
                            -1,
                        )
                    ),
                }
            )

            return result


# ============================================================
# One persistent sieve worker per physical GPU
# ============================================================


def _sieve_worker_main(
    physical_gpu_id: int,
    request_queue,
    response_queues,
):
    """Own one physical GPU for the lifetime of A11."""

    signal.signal(
        signal.SIGINT,
        signal.SIG_IGN,
    )

    # GPU affinity must be established before importing the pybind module.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)

    os.environ["A11_PHYSICAL_GPU_ID"] = str(physical_gpu_id)

    os.environ.pop(
        "LATTICE_DISABLE_CUDA",
        None,
    )

    gpu_workdir = Path(SIEVE_WORKDIR) / f"gpu{physical_gpu_id}"

    gpu_workdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    os.environ["LATTICE_SIEVE_WORKDIR"] = str(gpu_workdir)

    os.environ["LATTICE_SIEVE_KEEP_WORKDIR"] = str(int(SIEVE_KEEP_WORKDIR))
    # Native bridge consumes the positive cleanup flag, not KEEP_WORKDIR.
    os.environ["LATTICE_SIEVE_CLEANUP"] = str(int(not bool(SIEVE_KEEP_WORKDIR)))

    # Native code may also read these defaults when invoked outside
    # sieve_reduce_serialized().
    os.environ["A11_SIEVE_MAX_CANDIDATES"] = str(SIEVE_MAX_CANDIDATES)

    os.environ["A11_SIEVE_MAX_ROUNDS"] = str(SIEVE_MAX_ROUNDS)

    os.environ["A11_SIEVE_MAX_PAIRS"] = str(SIEVE_MAX_PAIRS)

    os.environ["A11_SIEVE_TIME_BUDGET_S"] = str(SIEVE_TIME_BUDGET_S)

    os.environ["A11_SIEVE_MEMORY_BUDGET_MB"] = str(SIEVE_MEMORY_BUDGET_MB)

    os.environ["A11_SIEVE_FREE_DIM"] = str(SIEVE_FREE_DIM)

    os.environ["A11_SIEVE_FREE_DIM_CAP"] = str(SIEVE_FREE_DIM_CAP)

    for path in (
        PROJECT_ROOT,
        BACKEND_DIR,
        BACKEND_BUILD_DIR,
    ):
        if path not in sys.path:
            sys.path.insert(
                0,
                path,
            )

    import my_project_backend

    required = (
        "backend_api_version",
        "sieve_reduce_serialized",
        "cuda_available",
        "shutdown_backend",
    )

    missing = [
        name
        for name in required
        if not hasattr(
            my_project_backend,
            name,
        )
    ]

    if missing:
        raise RuntimeError(
            "Persistent GPU sieve backend is missing:\n  " + "\n  ".join(missing)
        )

    api_version = int(my_project_backend.backend_api_version())
    if api_version != BACKEND_API_VERSION:
        raise RuntimeError(
            f"GPU{physical_gpu_id}: native backend API version {api_version} "
            f"!= expected {BACKEND_API_VERSION}; rebuild Backend/build"
        )

    if not bool(my_project_backend.cuda_available()):
        raise RuntimeError(
            f"GPU{physical_gpu_id}: native CUDA backend is not available"
        )

    if hasattr(my_project_backend, "shutdown_backend"):
        atexit.register(my_project_backend.shutdown_backend)

    completed_tasks = 0
    while True:
        task = request_queue.get()

        if task is None:
            break

        command = task.get("cmd")

        if command == "close":
            break

        if command != "sieve":
            continue

        env_id = int(task["env_id"])

        task_id = int(task["task_id"])

        response = {
            "task_id": task_id,
            "worker_pid": os.getpid(),
            "ok": False,
        }

        # Acknowledge dequeue before entering the native call.  This separates
        # legitimate queueing behind the other envs on this GPU from a native
        # execution that has actually wedged.
        try:
            response_queues[env_id].put(
                {
                    "task_id": task_id,
                    "worker_pid": os.getpid(),
                    "status": "started",
                    "ok": False,
                },
                timeout=max(0.1, SIEVE_RESPONSE_PUT_TIMEOUT_S),
            )
        except Exception:
            pass

        print(
            f"[sieve] START env={env_id} gpu={physical_gpu_id} "
            f"beta={int(task['beta'])}",
            flush=True,
        )

        try:
            budget = dict(
                task.get(
                    "budget",
                    {},
                )
            )

            with _silence_native_output():
                raw = my_project_backend.sieve_reduce_serialized(
                    str(task["block_matrix"]),
                    int(task["beta"]),
                    int(
                        budget.get(
                            "max_candidates",
                            SIEVE_MAX_CANDIDATES,
                        )
                    ),
                    int(
                        budget.get(
                            "max_rounds",
                            SIEVE_MAX_ROUNDS,
                        )
                    ),
                    int(
                        budget.get(
                            "max_pairs",
                            SIEVE_MAX_PAIRS,
                        )
                    ),
                    float(
                        budget.get(
                            "time_budget_s",
                            SIEVE_TIME_BUDGET_S,
                        )
                    ),
                    int(
                        budget.get(
                            "memory_budget_mb",
                            SIEVE_MEMORY_BUDGET_MB,
                        )
                    ),
                    float(
                        budget.get(
                            "min_b1_rel_improvement",
                            SIEVE_B1_REL_IMPROVEMENT,
                        )
                    ),
                    float(
                        budget.get(
                            "min_logpot_improvement",
                            SIEVE_LOGPOT_IMPROVEMENT,
                        )
                    ),
                    int(
                        budget.get(
                            "free_dim",
                            SIEVE_FREE_DIM,
                        )
                    ),
                    int(
                        budget.get(
                            "free_dim_cap",
                            SIEVE_FREE_DIM_CAP,
                        )
                    ),
                )

            result = dict(raw)
            response["ok"] = True
            response["result"] = result
            print(
                f"[sieve] END env={env_id} gpu={physical_gpu_id} "
                f"beta={int(task['beta'])} "
                f"shortened={_format_sieve_shortening(result)}",
                flush=True,
            )

        except BaseException as exc:
            response["error"] = f"{type(exc).__name__}: {exc}"

            response["traceback"] = traceback.format_exc()
            print(
                f"[sieve] END env={env_id} gpu={physical_gpu_id} "
                f"beta={int(task['beta'])} shortened=n/a "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )

        try:
            response_queues[env_id].put(
                response,
                timeout=max(0.1, SIEVE_RESPONSE_PUT_TIMEOUT_S),
            )
        except Exception:
            # The environment may have been terminated during shutdown.
            pass

        completed_tasks += 1
        if SIEVE_WORKER_MAX_TASKS > 0 and completed_tasks >= SIEVE_WORKER_MAX_TASKS:
            break

    try:
        if hasattr(my_project_backend, "shutdown_backend"):
            my_project_backend.shutdown_backend()
    except Exception:
        pass



class PersistentSieveService:
    """Create and own exactly one process per physical GPU."""

    def __init__(
        self,
        env_count: int,
        gpu_ids=GPU_IDS,
    ):
        self.env_count = int(env_count)

        self.gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids)

        self._closed = False

        self.request_queues = {
            gpu_id: mp.Queue(maxsize=SIEVE_QUEUE_SIZE) for gpu_id in self.gpu_ids
        }

        # Each environment has exactly one outstanding action, so one
        # dedicated response queue is enough and avoids result routing races.
        self.response_queues = [mp.Queue(maxsize=4) for _ in range(self.env_count)]

        self.processes = []

        for gpu_id in self.gpu_ids:
            self.processes.append(self._spawn_gpu_worker(gpu_id))

    def _spawn_gpu_worker(self, gpu_id: int):
        process = mp.Process(
            target=_sieve_worker_main,
            args=(
                int(gpu_id),
                self.request_queues[int(gpu_id)],
                self.response_queues,
            ),
            daemon=False,
            name=f"a11-sieve-gpu{int(gpu_id)}",
        )
        process.start()
        return process

    def close(self):
        if self._closed:
            return

        self._closed = True

        for gpu_id in self.gpu_ids:
            try:
                self.request_queues[gpu_id].put_nowait(
                    {
                        "cmd": "close",
                    }
                )
            except Exception:
                pass

        deadline = time.monotonic() + max(
            0.0,
            SIEVE_SERVICE_CLOSE_SECONDS,
        )

        for process in self.processes:
            remaining = max(
                0.0,
                deadline - time.monotonic(),
            )

            process.join(timeout=remaining)

        survivors = [process for process in self.processes if process.is_alive()]

        for process in survivors:
            try:
                process.terminate()
            except Exception:
                pass

        for process in survivors:
            process.join(timeout=3.0)

        survivors = [process for process in survivors if process.is_alive()]

        for process in survivors:
            try:
                process.kill()
            except Exception:
                pass

        for process in survivors:
            process.join(timeout=2.0)

        for queue in self.request_queues.values():
            try:
                queue.close()
            except Exception:
                pass

        for queue in self.response_queues:
            try:
                queue.close()
            except Exception:
                pass

    def assert_healthy(self):
        if self._closed:
            return

        # Request queues are process-independent. If a GPU owner exits (native
        # crash or planned allocator recycle), replacing only that owner keeps
        # queued requests intact. The one request already dequeued by a crashed
        # worker is recovered by the env-level wall-clock timeout/restart.
        for index, gpu_id in enumerate(self.gpu_ids):
            process = self.processes[index]
            if process.is_alive() or process.exitcode is None:
                continue
            exitcode = process.exitcode
            pid = process.pid
            process.join(timeout=0.2)
            replacement = self._spawn_gpu_worker(gpu_id)
            self.processes[index] = replacement
            print(
                f"[sieve-service] restarted GPU{gpu_id} worker: "
                f"old_pid={pid}, exitcode={exitcode}, new_pid={replacement.pid}",
                flush=True,
            )


# ============================================================
# CPU-only environment process
# ============================================================


def env_worker(
    remote,
    parent_remote,
    filepath: str,
    env_id: int,
    physical_gpu_id: int,
    cpu_gate,
    sieve_request_queue,
    sieve_response_queue,
    generation: int = 0,
):
    import faulthandler

    # Only the main process handles Ctrl+C.
    signal.signal(
        signal.SIGINT,
        signal.SIG_IGN,
    )

    # Environment processes must not create CUDA contexts or Pool_hd objects.
    #
    # They own exact MPZ matrices and run CPU-side BKZ/enumeration only.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["LATTICE_DISABLE_CUDA"] = "1"

    # This records which persistent GPU worker receives this env's sieve jobs.
    os.environ["A11_PHYSICAL_GPU_ID"] = str(physical_gpu_id)
    os.environ["LATTICE_ENUM_MAX_SECONDS"] = str(max(0.0, ENUM_TIME_BUDGET_S))

    cpu_affinity = configure_env_runtime(env_id)

    # Import backend only after disabling CUDA in this child process.
    from .backend import (
        clear_process_sieve_client,
        set_process_sieve_client,
    )

    sieve_client = SieveClient(
        request_queue=sieve_request_queue,
        response_queue=sieve_response_queue,
        env_id=env_id,
        gpu_id=physical_gpu_id,
        generation=generation,
    )

    set_process_sieve_client(sieve_client)

    # environment.py can remain unchanged. Its LatticeBackend constructor
    # reads the process-local client registered above.
    from .environment import LatticeEnv

    faulthandler.enable(all_threads=True)

    parent_remote.close()

    env = None
    last_cmd = None
    last_action = None
    current_file = filepath

    try:
        print(
            f"[env{env_id}] "
            f"pid={os.getpid()} "
            f"cpu_only=1 "
            f"sieve_gpu={physical_gpu_id} "
            f"generation={generation} "
            f"cpu_affinity={list(cpu_affinity)}",
            flush=True,
        )

        env = LatticeEnv(
            current_file,
            env_id=env_id,
            cpu_gate=cpu_gate,
            gpu_gate=None,
            global_gpu_gate=None,
            gpu_id=physical_gpu_id,
        )

        while True:
            try:
                last_cmd, data = remote.recv()
            except EOFError:
                break

            if last_cmd == "step":
                action_idx = int(data)

                pos, beta = env.action_list[action_idx]

                last_action = {
                    "action_idx": action_idx,
                    "pos": pos,
                    "beta": beta,
                    "pool_id": (env.current_pool_id),
                    "step": (env.current_step),
                    "sieve_gpu": (physical_gpu_id),
                }

                state, reward, done, info = env.step(action_idx)

                best_update = env.pop_best_update()

                if best_update is not None:
                    info["best_update"] = best_update

                remote.send(
                    (
                        state,
                        reward,
                        done,
                        info,
                    )
                )

            elif last_cmd == "reset":
                remote.send(env.reset())

            elif last_cmd == "load":
                next_file = str(data)

                env.close()

                current_file = next_file

                env = LatticeEnv(
                    current_file,
                    env_id=env_id,
                    cpu_gate=cpu_gate,
                    gpu_gate=None,
                    global_gpu_gate=None,
                    gpu_id=physical_gpu_id,
                )

                last_action = None

                remote.send(env.reset())

            elif last_cmd == "get_best":
                remote.send(env.get_best_payload())

            elif last_cmd == "close":
                break

            else:
                raise RuntimeError(f"Unknown command in env{env_id}: {last_cmd!r}")

    except (
        EOFError,
        BrokenPipeError,
    ):
        pass

    except Exception as exc:
        dim, seed_id = parse_dim_seed(current_file)

        print(
            "\n"
            f"[env{env_id}] FATAL\n"
            f"  pid         = {os.getpid()}\n"
            f"  sieve_gpu   = {physical_gpu_id}\n"
            f"  dim         = {dim}\n"
            f"  seed        = {seed_id}\n"
            f"  file        = {current_file}\n"
            f"  last_cmd    = {last_cmd!r}\n"
            f"  last_action = {last_action!r}\n"
            f"  exception   = "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )

        traceback.print_exc()

        raise

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        clear_process_sieve_client()

        try:
            remote.close()
        except Exception:
            pass


# ============================================================
# Vector environment
# ============================================================


class SubprocVecEnv:
    def __init__(
        self,
        files,
        env_count: int = ENV_COUNT,
        envs_per_file: int = ENVS_PER_FILE,
    ):
        self.dataset_files = list(files)
        if not self.dataset_files:
            raise ValueError("No dataset files were provided.")

        ordered = list(self.dataset_files)
        random.Random(SEED).shuffle(ordered)
        jobs = [filepath for filepath in ordered for _ in range(envs_per_file)]
        if len(jobs) < env_count:
            raise ValueError(
                f"ENV_COUNT={env_count} requires at least {env_count} file jobs, "
                f"but only {len(jobs)} are available "
                f"({len(ordered)} files × {envs_per_file})."
            )

        self.num_envs = int(env_count)
        self._closed = False
        self._job_queue = deque(jobs)
        self.files = [self._job_queue.popleft() for _ in range(self.num_envs)]
        self.env_dims = [parse_dim_seed(filepath)[0] for filepath in self.files]
        self.env_seed_ids = [parse_dim_seed(filepath)[1] for filepath in self.files]
        self.dataset_pairs = sorted(
            {parse_dim_seed(filepath) for filepath in self.dataset_files}
        )
        self.dataset_dims = sorted({dim for dim, _ in self.dataset_pairs})

        # CPU scheduler remains a process-shared semaphore.
        self.cpu_gate = create_cpu_gate()

        # Exactly one persistent owner process per physical GPU.
        self.sieve_service = PersistentSieveService(
            env_count=self.num_envs,
            gpu_ids=GPU_IDS,
        )
        self.env_gpu_ids = [
            GPU_IDS[env_id % len(GPU_IDS)] for env_id in range(self.num_envs)
        ]
        self.gpu_assignment_counts = {
            gpu_id: self.env_gpu_ids.count(gpu_id) for gpu_id in GPU_IDS
        }

        # Connections are mutable because a dead native worker must be replaceable
        # without rebuilding the entire vector environment.
        self.remotes = [None] * self.num_envs
        self.processes = [None] * self.num_envs
        self.generations = [0] * self.num_envs
        for env_id, filepath in enumerate(self.files):
            self._spawn_env_process(env_id, filepath)

    # --------------------------------------------------------
    # Process lifecycle / fault isolation
    # --------------------------------------------------------

    def _spawn_env_process(self, env_id: int, filepath: str) -> None:
        if self._closed:
            raise RuntimeError("cannot spawn an environment after close()")

        remote, work_remote = mp.Pipe()
        physical_gpu_id = self.env_gpu_ids[env_id]
        generation = self.generations[env_id]
        process = mp.Process(
            target=env_worker,
            args=(
                work_remote,
                remote,
                filepath,
                env_id,
                physical_gpu_id,
                self.cpu_gate,
                self.sieve_service.request_queues[physical_gpu_id],
                self.sieve_service.response_queues[env_id],
                generation,
            ),
            daemon=True,
            name=f"a11-env-{env_id}-g{generation}",
        )
        process.start()
        work_remote.close()
        self.remotes[env_id] = remote
        self.processes[env_id] = process

    def _failure_message(self, env_id: int, prefix: str) -> str:
        process = self.processes[env_id]
        return (
            f"{prefix}:\n"
            f"  env_id       = {env_id}\n"
            f"  generation   = {self.generations[env_id]}\n"
            f"  dim          = {self.env_dims[env_id]}\n"
            f"  seed         = {self.env_seed_ids[env_id]}\n"
            f"  file         = {self.files[env_id]}\n"
            f"  sieve_gpu    = {self.env_gpu_ids[env_id]}\n"
            f"  worker_pid   = {getattr(process, 'pid', None)}\n"
            f"  worker_alive = {bool(process and process.is_alive())}\n"
            f"  worker_exit  = {getattr(process, 'exitcode', None)}\n"
            "  exit meanings: 1=Python exception, -6=SIGABRT, "
            "-9=SIGKILL/OOM, -11=SIGSEGV"
        )

    @staticmethod
    def _stop_process(process) -> None:
        if process is None:
            return
        if process.is_alive():
            try:
                process.terminate()
            except Exception:
                pass
            process.join(timeout=WORKER_TERMINATE_GRACE_SECONDS)
        if process.is_alive():
            try:
                process.kill()
            except Exception:
                pass
            process.join(timeout=WORKER_KILL_GRACE_SECONDS)
        else:
            process.join(timeout=0.1)

    def _recv_with_timeout(
        self,
        env_id: int,
        timeout_s: float,
        operation: str,
    ):
        remote = self.remotes[env_id]
        process = self.processes[env_id]
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        while True:
            if remote.poll(timeout=min(0.25, max(0.0, deadline - time.monotonic()))):
                try:
                    return remote.recv()
                except EOFError as exc:
                    raise RuntimeError(
                        self._failure_message(
                            env_id,
                            f"Environment worker exited during {operation}",
                        )
                    ) from exc
            if not process.is_alive() and process.exitcode is not None:
                raise RuntimeError(
                    self._failure_message(
                        env_id,
                        f"Environment worker exited during {operation}",
                    )
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    self._failure_message(
                        env_id,
                        f"Environment worker timed out during {operation} "
                        f"after {timeout_s:.1f}s",
                    )
                )

    def restart_one(self, env_id: int, *, filepath: str | None = None):
        """Replace exactly one environment and reset the same lattice file.

        The interrupted episode is intentionally discarded by trainer.py. The
        new generation gets a disjoint sieve task-id range, so a delayed response
        from the dead process cannot be mistaken for the new process's response.
        """
        if self._closed:
            raise RuntimeError("cannot restart an environment after close()")

        target_file = str(filepath or self.files[env_id])
        old_remote = self.remotes[env_id]
        old_process = self.processes[env_id]
        self._stop_process(old_process)
        try:
            old_remote.close()
        except Exception:
            pass

        self.files[env_id] = target_file
        self.env_dims[env_id], self.env_seed_ids[env_id] = parse_dim_seed(target_file)
        last_error = None
        max_attempts = max(1, int(ENV_MAX_CONSECUTIVE_FAILURES) + 1)

        for attempt in range(1, max_attempts + 1):
            if self.generations[env_id] >= 0xFFFF:
                raise RuntimeError(
                    f"env{env_id} restart generation exhausted; refusing task-id reuse"
                )
            self.generations[env_id] += 1
            self._spawn_env_process(env_id, target_file)
            try:
                self.remotes[env_id].send(("reset", None))
                return self._recv_with_timeout(
                    env_id,
                    ENV_RESTART_TIMEOUT_S,
                    f"restart/reset attempt {attempt}/{max_attempts}",
                )
            except Exception as exc:
                last_error = exc
                failed_remote = self.remotes[env_id]
                failed_process = self.processes[env_id]
                self._stop_process(failed_process)
                try:
                    failed_remote.close()
                except Exception:
                    pass
                if attempt < max_attempts:
                    print(
                        f"[A11] env{env_id} restart attempt {attempt} failed; retrying: {exc}",
                        flush=True,
                    )

        raise RuntimeError(
            f"env{env_id} could not be restarted after {max_attempts} attempts"
        ) from last_error

    # --------------------------------------------------------
    # Communication
    # --------------------------------------------------------

    def reset_all(self):
        # Start all expensive LLL+BKZ20 initializations in parallel as before.
        sent = [False] * self.num_envs
        for env_id, remote in enumerate(self.remotes):
            process = self.processes[env_id]
            if not process.is_alive():
                continue
            try:
                remote.send(("reset", None))
                sent[env_id] = True
            except (BrokenPipeError, EOFError, OSError):
                sent[env_id] = False

        states = [None] * self.num_envs
        for env_id in range(self.num_envs):
            try:
                if not sent[env_id]:
                    raise RuntimeError(
                        self._failure_message(
                            env_id,
                            "Environment worker unavailable during initial reset",
                        )
                    )
                states[env_id] = self._recv_with_timeout(
                    env_id,
                    ENV_RESTART_TIMEOUT_S,
                    "initial reset",
                )
            except Exception as exc:
                print(
                    f"[A11] env{env_id} initial reset failed; rebuilding slot: {exc}",
                    flush=True,
                )
                states[env_id] = self.restart_one(
                    env_id,
                    filepath=self.files[env_id],
                )
        return states

    def rotate_one(self, env_id: int):
        current_file = self.files[env_id]
        self._job_queue.append(current_file)
        next_file = self._job_queue.popleft()

        try:
            self.remotes[env_id].send(("load", next_file))
            state = self._recv_with_timeout(
                env_id,
                ENV_RESTART_TIMEOUT_S,
                "file rotation/preprocess",
            )
        except Exception as load_error:
            # Preserve the original round-robin assignment: if the worker dies
            # while preprocessing next_file, replace the slot directly on that
            # next file instead of silently running current_file for an extra
            # episode. The current file is already back at the queue tail.
            try:
                return self.restart_one(env_id, filepath=next_file)
            except Exception as restart_error:
                # Neither load nor replacement succeeded. Restore queue/file
                # ownership exactly to its pre-rotation state before surfacing.
                self._job_queue.appendleft(next_file)
                try:
                    self._job_queue.remove(current_file)
                except ValueError:
                    pass
                self.files[env_id] = current_file
                self.env_dims[env_id], self.env_seed_ids[env_id] = parse_dim_seed(
                    current_file
                )
                raise RuntimeError(
                    f"env{env_id} rotation failed and replacement on next file failed"
                ) from restart_error

        self.files[env_id] = next_file
        self.env_dims[env_id], self.env_seed_ids[env_id] = parse_dim_seed(next_file)
        return state

    def send_one(self, env_id: int, action: int):
        process = self.processes[env_id]
        if not process.is_alive():
            raise RuntimeError(
                self._failure_message(env_id, "Cannot dispatch to dead environment worker")
            )
        try:
            self.remotes[env_id].send(("step", action))
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RuntimeError(
                self._failure_message(env_id, "Environment dispatch failed")
            ) from exc

    def recv_one(self, env_id: int):
        try:
            return self.remotes[env_id].recv()
        except EOFError as exc:
            process = self.processes[env_id]
            process.join(timeout=0.5)
            raise RuntimeError(
                self._failure_message(env_id, "Environment worker exited unexpectedly")
            ) from exc

    def poll_ready(self, env_ids):
        """Return (ready_ids, failed_ids) without turning one crash into global exit."""
        self.sieve_service.assert_healthy()
        ready = []
        failed = []
        for env_id in env_ids:
            remote = self.remotes[env_id]
            if remote.poll(timeout=0):
                ready.append(env_id)
                continue
            process = self.processes[env_id]
            if not process.is_alive() and process.exitcode is not None:
                failed.append(env_id)
        return ready, failed

    def get_bests(self):
        for env_id, remote in enumerate(self.remotes):
            process = self.processes[env_id]
            if not process.is_alive():
                raise RuntimeError(
                    self._failure_message(env_id, "Environment died before get_best")
                )
            remote.send(("get_best", None))
        return [
            self._recv_with_timeout(env_id, ENV_RESTART_TIMEOUT_S, "get_best")
            for env_id in range(self.num_envs)
        ]

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    @staticmethod
    def _join_until(processes, timeout_seconds: float) -> list:
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        live = [process for process in processes if process is not None]
        for process in live:
            remaining = max(0.0, deadline - time.monotonic())
            process.join(timeout=remaining)
        return [process for process in live if process.is_alive()]

    def close(self):
        if self._closed:
            return
        self._closed = True

        try:
            # Phase 1: ask env processes to close cooperatively.
            for remote, process in zip(self.remotes, self.processes):
                if process is None or not process.is_alive():
                    continue
                try:
                    remote.send(("close", None))
                except Exception:
                    pass

            survivors = self._join_until(
                self.processes,
                WORKER_CLOSE_GRACE_SECONDS,
            )

            # Phase 2: terminate envs still waiting on Python/native/IPC.
            for process in survivors:
                try:
                    process.terminate()
                except Exception:
                    pass
            survivors = self._join_until(
                survivors,
                WORKER_TERMINATE_GRACE_SECONDS,
            )

            # Phase 3: hard-stop remaining environment processes.
            for process in survivors:
                try:
                    process.kill()
                except Exception:
                    pass
            self._join_until(survivors, WORKER_KILL_GRACE_SECONDS)

            for remote in self.remotes:
                if remote is None:
                    continue
                try:
                    remote.close()
                except Exception:
                    pass
        finally:
            # Stop GPU workers only after env processes can no longer submit work.
            self.sieve_service.close()

