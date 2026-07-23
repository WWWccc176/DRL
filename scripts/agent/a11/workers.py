from __future__ import annotations

import multiprocessing as mp
import os
import random
import signal
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from queue import Empty
from typing import Any

from .config import (
    BACKEND_BUILD_DIR,
    BACKEND_DIR,
    ENV_COUNT,
    ENVS_PER_FILE,
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
    SIEVE_QUEUE_SIZE,
    SIEVE_RESPONSE_POLL_SECONDS,
    SIEVE_SERVICE_CLOSE_SECONDS,
    SIEVE_TIME_BUDGET_S,
    SIEVE_WORKDIR,
    WORKER_CLOSE_GRACE_SECONDS,
    WORKER_KILL_GRACE_SECONDS,
    WORKER_TERMINATE_GRACE_SECONDS,
)
from .io_utils import parse_dim_seed
from .runtime import configure_env_runtime
from .scheduler import create_cpu_gate


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
    ):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.env_id = int(env_id)
        self.gpu_id = int(gpu_id)
        self._counter = 0

    def _next_task_id(self) -> int:
        self._counter += 1

        # High bits identify the environment. Low bits are the per-env sequence.
        return (self.env_id << 48) | self._counter

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

        self.request_queue.put(request)

        while True:
            try:
                response = self.response_queue.get(
                    timeout=max(
                        0.1,
                        SIEVE_RESPONSE_POLL_SECONDS,
                    )
                )
            except Empty:
                # A beta=95 sieve may legitimately run for a long time.
                # The parent shutdown path can still terminate this env process.
                continue

            received_task_id = int(
                response.get(
                    "task_id",
                    -1,
                )
            )

            if received_task_id != task_id:
                raise RuntimeError(
                    "Persistent sieve response mismatch:\n"
                    f"  env_id   = {self.env_id}\n"
                    f"  gpu_id   = {self.gpu_id}\n"
                    f"  expected = {task_id}\n"
                    f"  received = {received_task_id}"
                )

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
        "sieve_reduce_serialized",
        "cuda_available",
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

    if not bool(my_project_backend.cuda_available()):
        raise RuntimeError(
            f"GPU{physical_gpu_id}: native CUDA backend is not available"
        )

    print(
        f"[sieve-worker] "
        f"pid={os.getpid()} "
        f"physical_gpu={physical_gpu_id} "
        f"logical_cuda=0 "
        f"memory_budget_mb={SIEVE_MEMORY_BUDGET_MB} "
        f"workdir={gpu_workdir}",
        flush=True,
    )

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

        try:
            budget = dict(
                task.get(
                    "budget",
                    {},
                )
            )

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

            response["ok"] = True
            response["result"] = dict(raw)

        except BaseException as exc:
            response["error"] = f"{type(exc).__name__}: {exc}"

            response["traceback"] = traceback.format_exc()

        try:
            response_queues[env_id].put(response)
        except Exception:
            # The environment may have been terminated during shutdown.
            pass

    print(
        f"[sieve-worker] physical_gpu={physical_gpu_id} pid={os.getpid()} stopped",
        flush=True,
    )


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
            process = mp.Process(
                target=_sieve_worker_main,
                args=(
                    gpu_id,
                    self.request_queues[gpu_id],
                    self.response_queues,
                ),
                daemon=False,
                name=(f"a11-sieve-gpu{gpu_id}"),
            )

            process.start()

            self.processes.append(process)

    def close(self):
        if self._closed:
            return

        self._closed = True

        for gpu_id in self.gpu_ids:
            try:
                self.request_queues[gpu_id].put(
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
                f"ENV_COUNT={env_count} requires at least "
                f"{env_count} file jobs, but only "
                f"{len(jobs)} are available "
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

        # Create GPU owners before creating environment processes.
        #
        # GPU worker count:
        #     exactly len(GPU_IDS)
        #
        # At any instant:
        #     GPU0 <= 1 sieve
        #     GPU1 <= 1 sieve
        #     GPU2 <= 1 sieve
        #     GPU3 <= 1 sieve
        #
        # There is no global <=2 sieve restriction.
        self.sieve_service = PersistentSieveService(
            env_count=self.num_envs,
            gpu_ids=GPU_IDS,
        )

        # Each environment always submits to the same physical GPU queue.
        self.env_gpu_ids = [
            GPU_IDS[env_id % len(GPU_IDS)] for env_id in range(self.num_envs)
        ]

        self.gpu_assignment_counts = {
            gpu_id: (self.env_gpu_ids.count(gpu_id)) for gpu_id in GPU_IDS
        }

        self.remotes, self.work_remotes = zip(
            *[mp.Pipe() for _ in range(self.num_envs)]
        )

        self.processes = []

        for env_id, (
            work_remote,
            remote,
            filepath,
        ) in enumerate(
            zip(
                self.work_remotes,
                self.remotes,
                self.files,
            )
        ):
            physical_gpu_id = self.env_gpu_ids[env_id]

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
                ),
                daemon=True,
                name=f"a11-env-{env_id}",
            )

            process.start()

            self.processes.append(process)

        for work_remote in self.work_remotes:
            work_remote.close()

    # --------------------------------------------------------
    # Communication
    # --------------------------------------------------------

    def reset_all(self):
        for remote in self.remotes:
            remote.send(
                (
                    "reset",
                    None,
                )
            )

        return [remote.recv() for remote in self.remotes]

    def rotate_one(
        self,
        env_id: int,
    ):
        current_file = self.files[env_id]

        self._job_queue.append(current_file)

        next_file = self._job_queue.popleft()

        self.remotes[env_id].send(
            (
                "load",
                next_file,
            )
        )

        state = self.remotes[env_id].recv()

        self.files[env_id] = next_file

        (
            self.env_dims[env_id],
            self.env_seed_ids[env_id],
        ) = parse_dim_seed(next_file)

        return state

    def send_one(
        self,
        env_id: int,
        action: int,
    ):
        self.remotes[env_id].send(
            (
                "step",
                action,
            )
        )

    def recv_one(
        self,
        env_id: int,
    ):
        try:
            return self.remotes[env_id].recv()

        except EOFError as exc:
            process = self.processes[env_id]

            process.join(timeout=0.5)

            raise RuntimeError(
                "\nEnvironment worker exited unexpectedly:\n"
                f"  env_id       = {env_id}\n"
                f"  dim          = {self.env_dims[env_id]}\n"
                f"  seed         = {self.env_seed_ids[env_id]}\n"
                f"  file         = {self.files[env_id]}\n"
                f"  sieve_gpu    = {self.env_gpu_ids[env_id]}\n"
                f"  worker_pid   = {process.pid}\n"
                f"  worker_alive = {process.is_alive()}\n"
                f"  worker_exit  = {process.exitcode}\n"
                "  exit meanings: "
                "1=Python exception, "
                "-6=SIGABRT, "
                "-9=SIGKILL/OOM, "
                "-11=SIGSEGV\n"
            ) from exc

    def poll_ready(
        self,
        env_ids,
    ):
        return [env_id for env_id in env_ids if self.remotes[env_id].poll(timeout=0)]

    def get_bests(self):
        for remote in self.remotes:
            remote.send(
                (
                    "get_best",
                    None,
                )
            )

        return [remote.recv() for remote in self.remotes]

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    @staticmethod
    def _join_until(
        processes,
        timeout_seconds: float,
    ) -> list:
        deadline = time.monotonic() + max(
            0.0,
            timeout_seconds,
        )

        for process in processes:
            remaining = max(
                0.0,
                deadline - time.monotonic(),
            )

            process.join(timeout=remaining)

        return [process for process in processes if process.is_alive()]

    def close(self):
        if self._closed:
            return

        self._closed = True

        try:
            # Phase 1: ask env processes to close cooperatively.
            for remote, process in zip(
                self.remotes,
                self.processes,
            ):
                if not process.is_alive():
                    continue

                try:
                    remote.send(
                        (
                            "close",
                            None,
                        )
                    )
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

            self._join_until(
                survivors,
                WORKER_KILL_GRACE_SECONDS,
            )

            for remote in self.remotes:
                try:
                    remote.close()
                except Exception:
                    pass

        finally:
            # Stop GPU workers only after env processes can no longer submit
            # new sieve requests.
            self.sieve_service.close()
