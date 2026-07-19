from __future__ import annotations

import os
import random
import sys
from pathlib import Path

from .config import (
    ENV_CPU_THREADS,
    MAIN_CPU_THREADS,
    PROJECT_ROOT,
    SEED,
)


def _set_common_thread_env(omp_threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(omp_threads)))
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["OMP_PROC_BIND"] = "close"
    os.environ["OMP_PLACES"] = "threads"

    # Keep nested BLAS pools from multiplying the 48 environment processes.
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _read_int(path: Path, default: int) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return default


def _parse_cpu_list(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def physical_core_affinity_groups() -> list[tuple[int, ...]]:
    """Return one logical-CPU sibling group per physical core.

    Groups are interleaved across sockets so env0/env1/... do not fill one socket
    before using the other. On the target 2x24-core HT system this yields 48 groups,
    normally two logical CPUs per group.
    """
    try:
        allowed = set(os.sched_getaffinity(0))
    except Exception:
        allowed = set(range(os.cpu_count() or 1))

    topology = Path("/sys/devices/system/cpu")
    by_socket: dict[int, dict[int, set[int]]] = {}

    for cpu in sorted(allowed):
        cpu_dir = topology / f"cpu{cpu}" / "topology"
        package_id = _read_int(cpu_dir / "physical_package_id", 0)
        core_id = _read_int(cpu_dir / "core_id", cpu)

        siblings = {cpu}
        try:
            siblings = _parse_cpu_list(
                (cpu_dir / "thread_siblings_list").read_text(encoding="utf-8")
            )
            siblings &= allowed
            if not siblings:
                siblings = {cpu}
        except Exception:
            pass

        by_socket.setdefault(package_id, {}).setdefault(core_id, set()).update(siblings)

    socket_groups: list[list[tuple[int, ...]]] = []
    for package_id in sorted(by_socket):
        groups = [
            tuple(sorted(by_socket[package_id][core_id]))
            for core_id in sorted(by_socket[package_id])
        ]
        socket_groups.append(groups)

    if not socket_groups:
        return [(cpu,) for cpu in sorted(allowed)]

    # Round-robin sockets: socket0-core0, socket1-core0, socket0-core1, ...
    interleaved: list[tuple[int, ...]] = []
    max_len = max(len(groups) for groups in socket_groups)
    for i in range(max_len):
        for groups in socket_groups:
            if i < len(groups):
                interleaved.append(groups)

    # Deduplicate in case unusual topology exposes overlapping sibling lists.
    seen: set[tuple[int, ...]] = set()
    unique: list[tuple[int, ...]] = []
    for group in interleaved:
        key = tuple(sorted(set(group)))
        if key and key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def bind_env_cpu_affinity(env_id: int) -> tuple[int, ...]:
    groups = physical_core_affinity_groups()
    if not groups:
        return ()
    cpus = groups[int(env_id) % len(groups)]
    try:
        os.sched_setaffinity(0, set(cpus))
    except Exception:
        pass
    return cpus


def configure_main_runtime() -> None:
    _set_common_thread_env(MAIN_CPU_THREADS)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


def configure_env_runtime(env_id: int) -> tuple[int, ...]:
    # CUDA_VISIBLE_DEVICES is set by workers.py before this function is called.
    # The backend therefore sees exactly one assigned GPU as logical cuda:0.
    _set_common_thread_env(ENV_CPU_THREADS)
    cpus = bind_env_cpu_affinity(env_id)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    return cpus


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)

    # Import NumPy only after configure_*_runtime() has set BLAS/OpenMP variables.
    import numpy as np

    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_device():
    import torch

    torch.set_num_threads(MAIN_CPU_THREADS)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def status(msg: str) -> None:
    sys.stdout.write("\r\033[K" + msg)
    sys.stdout.flush()


def log(msg: str) -> None:
    sys.stdout.write("\r\033[K" + msg + "\n")
    sys.stdout.flush()
