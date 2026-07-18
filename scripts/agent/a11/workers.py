from __future__ import annotations

import multiprocessing as mp
import os
import random
import sys
import traceback
from collections import deque

from .config import ENV_COUNT, ENVS_PER_FILE, SEED
from .io_utils import parse_dim_seed
from .runtime import configure_env_runtime
from .scheduler import create_reduction_gate


def env_worker(remote, parent_remote, filepath: str, env_id: int, reduction_gate):
    import faulthandler

    configure_env_runtime()
    from .environment import LatticeEnv

    faulthandler.enable(all_threads=True)
    parent_remote.close()
    env = None
    last_cmd = None
    last_action = None
    current_file = filepath

    try:
        env = LatticeEnv(current_file, env_id=env_id, reduction_gate=reduction_gate)
        while True:
            last_cmd, data = remote.recv()

            if last_cmd == "step":
                action_idx = int(data)
                pos, beta = env.action_list[action_idx]
                last_action = {
                    "action_idx": action_idx,
                    "pos": pos,
                    "beta": beta,
                    "pool_id": env.current_pool_id,
                    "step": env.current_step,
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
                env = LatticeEnv(current_file, env_id=env_id, reduction_gate=reduction_gate)
                last_action = None
                remote.send(env.reset())

            elif last_cmd == "get_best":
                remote.send(env.get_best_payload())

            elif last_cmd == "close":
                break

            else:
                raise RuntimeError(f"Unknown command in env{env_id}: {last_cmd!r}")

    except BaseException as exc:
        dim, seed_id = parse_dim_seed(current_file)
        print(
            "\n"
            f"[env{env_id}] FATAL\n"
            f"  pid         = {os.getpid()}\n"
            f"  dim         = {dim}\n"
            f"  seed        = {seed_id}\n"
            f"  file        = {current_file}\n"
            f"  last_cmd    = {last_cmd!r}\n"
            f"  last_action = {last_action!r}\n"
            f"  exception   = {type(exc).__name__}: {exc}",
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
        self._job_queue = deque(jobs)
        self.files = [self._job_queue.popleft() for _ in range(self.num_envs)]
        self.env_dims = [parse_dim_seed(f)[0] for f in self.files]
        self.env_seed_ids = [parse_dim_seed(f)[1] for f in self.files]
        self.dataset_pairs = sorted({parse_dim_seed(f) for f in self.dataset_files})
        self.dataset_dims = sorted({dim for dim, _ in self.dataset_pairs})

        self.reduction_gate = create_reduction_gate()

        self.remotes, self.work_remotes = zip(
            *[mp.Pipe() for _ in range(self.num_envs)]
        )
        self.processes = []
        for env_id, (work_remote, remote, filepath) in enumerate(
            zip(self.work_remotes, self.remotes, self.files)
        ):
            process = mp.Process(
                target=env_worker,
                args=(work_remote, remote, filepath, env_id, self.reduction_gate),
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
        """Finish one file-cycle and assign this fixed env slot to the next file job."""
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

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
