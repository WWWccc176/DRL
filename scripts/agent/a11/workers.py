from __future__ import annotations

import multiprocessing as mp
import os
import random
import signal
import sys
import time
import traceback
from collections import deque

from .config import (
    ENV_COUNT,
    ENVS_PER_FILE,
    GPU_IDS,
    SEED,
    WORKER_CLOSE_GRACE_SECONDS,
    WORKER_KILL_GRACE_SECONDS,
    WORKER_TERMINATE_GRACE_SECONDS,
)
from .io_utils import parse_dim_seed
from .runtime import configure_env_runtime
from .scheduler import create_reduction_gates


def env_worker(
    remote,
    parent_remote,
    filepath: str,
    env_id: int,
    physical_gpu_id: int,
    cpu_gate,
    gpu_gate,
    global_gpu_gate,
):
    import faulthandler

    # Only the main process handles Ctrl+C. Workers may be inside pybind/CUDA calls,
    # so propagating SIGINT to all 48 workers creates traceback storms and slow exit.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # GPU affinity MUST be fixed before importing environment/backend/native .so.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    os.environ["A11_PHYSICAL_GPU_ID"] = str(physical_gpu_id)

    cpu_affinity = configure_env_runtime(env_id)
    from .environment import LatticeEnv

    faulthandler.enable(all_threads=True)
    parent_remote.close()
    env = None
    last_cmd = None
    last_action = None
    current_file = filepath

    try:
        print(
            f"[env{env_id}] pid={os.getpid()} "
            f"physical_gpu={physical_gpu_id} logical_cuda=0 "
            f"cpu_affinity={list(cpu_affinity)}",
            flush=True,
        )

        env = LatticeEnv(
            current_file,
            env_id=env_id,
            cpu_gate=cpu_gate,
            gpu_gate=gpu_gate,
            global_gpu_gate=global_gpu_gate,
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
                    "pool_id": env.current_pool_id,
                    "step": env.current_step,
                    "physical_gpu": physical_gpu_id,
                }
                state, reward, done, info = env.step(action_idx)
                best_update = env.pop_best_update()
                if best_update is not None:
                    info["best_update"] = best_update
                remote.send((state, reward, done, info))

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
                    gpu_gate=gpu_gate,
                    global_gpu_gate=global_gpu_gate,
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

    except (EOFError, BrokenPipeError):
        # Parent is shutting down or the IPC pipe has already been closed.
        pass
    except Exception as exc:
        dim, seed_id = parse_dim_seed(current_file)
        print(
            "\n"
            f"[env{env_id}] FATAL\n"
            f"  pid          = {os.getpid()}\n"
            f"  physical_gpu = {physical_gpu_id}\n"
            f"  dim          = {dim}\n"
            f"  seed         = {seed_id}\n"
            f"  file         = {current_file}\n"
            f"  last_cmd     = {last_cmd!r}\n"
            f"  last_action  = {last_action!r}\n"
            f"  exception    = {type(exc).__name__}: {exc}",
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
        try:
            remote.close()
        except Exception:
            pass


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
                f"but {len(jobs)} are available ({len(ordered)} files × {envs_per_file})."
            )

        self.num_envs = env_count
        self._closed = False
        self._job_queue = deque(jobs)
        self.files = [self._job_queue.popleft() for _ in range(self.num_envs)]
        self.env_dims = [parse_dim_seed(f)[0] for f in self.files]
        self.env_seed_ids = [parse_dim_seed(f)[1] for f in self.files]
        self.dataset_pairs = sorted({parse_dim_seed(f) for f in self.dataset_files})
        self.dataset_dims = sorted({dim for dim, _ in self.dataset_pairs})

        (
            self.cpu_gate,
            self.global_gpu_gate,
            self.gpu_gates,
        ) = create_reduction_gates()

        self.env_gpu_ids = [
            GPU_IDS[env_id % len(GPU_IDS)] for env_id in range(self.num_envs)
        ]
        self.gpu_assignment_counts = {
            gpu_id: self.env_gpu_ids.count(gpu_id) for gpu_id in GPU_IDS
        }

        self.remotes, self.work_remotes = zip(
            *[mp.Pipe() for _ in range(self.num_envs)]
        )
        self.processes = []

        for env_id, (work_remote, remote, filepath) in enumerate(
            zip(self.work_remotes, self.remotes, self.files)
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
                    self.gpu_gates[physical_gpu_id],
                    self.global_gpu_gate,
                ),
                daemon=True,
            )
            process.start()
            self.processes.append(process)

        for work_remote in self.work_remotes:
            work_remote.close()

    def reset_all(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return [remote.recv() for remote in self.remotes]

    def rotate_one(self, env_id: int):
        current_file = self.files[env_id]
        self._job_queue.append(current_file)
        next_file = self._job_queue.popleft()

        self.remotes[env_id].send(("load", next_file))
        state = self.remotes[env_id].recv()

        self.files[env_id] = next_file
        self.env_dims[env_id], self.env_seed_ids[env_id] = parse_dim_seed(next_file)
        return state

    def send_one(self, env_id: int, action: int):
        self.remotes[env_id].send(("step", action))

    def recv_one(self, env_id: int):
        try:
            return self.remotes[env_id].recv()
        except EOFError as exc:
            proc = self.processes[env_id]
            proc.join(timeout=0.5)
            raise RuntimeError(
                "\nEnvironment worker exited unexpectedly:\n"
                f"  env_id        = {env_id}\n"
                f"  dim           = {self.env_dims[env_id]}\n"
                f"  seed          = {self.env_seed_ids[env_id]}\n"
                f"  file          = {self.files[env_id]}\n"
                f"  worker_pid    = {proc.pid}\n"
                f"  worker_alive  = {proc.is_alive()}\n"
                f"  worker_exit   = {proc.exitcode}  "
                "(1=Py异常  -6=SIGABRT  -9=SIGKILL/OOM  -11=SIGSEGV)\n"
            ) from exc

    def poll_ready(self, env_ids):
        return [env_id for env_id in env_ids if self.remotes[env_id].poll(timeout=0)]

    @staticmethod
    def _join_until(processes, timeout_seconds: float) -> list:
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        for process in processes:
            remaining = max(0.0, deadline - time.monotonic())
            process.join(timeout=remaining)
        return [process for process in processes if process.is_alive()]

    def close(self):
        if self._closed:
            return
        self._closed = True

        # Phase 1: cooperative shutdown. Busy workers may not read the command until
        # a native BKZ/sieve call returns, so this phase is explicitly time-bounded.
        for remote, process in zip(self.remotes, self.processes):
            if not process.is_alive():
                continue
            try:
                remote.send(("close", None))
            except Exception:
                pass

        survivors = self._join_until(
            self.processes,
            WORKER_CLOSE_GRACE_SECONDS,
        )

        # Phase 2: terminate workers still blocked in Python/native code.
        for process in survivors:
            try:
                process.terminate()
            except Exception:
                pass

        survivors = self._join_until(
            survivors,
            WORKER_TERMINATE_GRACE_SECONDS,
        )

        # Phase 3: final hard stop. At this point the whole vector environment is
        # shutting down, so preserving a stuck native CUDA call is less important
        # than releasing host RAM and restoring the server.
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
