from __future__ import annotations

import multiprocessing as mp
from contextlib import contextmanager

from .config import CPU_REDUCTION_CONCURRENCY


def create_cpu_gate():
    """Create the global gate for CPU-side native reductions."""
    return mp.BoundedSemaphore(CPU_REDUCTION_CONCURRENCY)


def create_reduction_gates():
    """Backward-compatible scheduler factory.

    GPU admission is no longer controlled by multiprocessing semaphores.

    Each physical GPU has one persistent sieve process, so:
        - one GPU cannot run two sieve calls simultaneously;
        - four GPUs can run four sieve calls simultaneously.

    The returned GPU values are retained only so older imports do not fail.
    """
    cpu_gate = create_cpu_gate()

    global_gpu_gate = None
    gpu_gates = {}

    return (
        cpu_gate,
        global_gpu_gate,
        gpu_gates,
    )


@contextmanager
def reduction_slot(gate):
    """Acquire one CPU reduction slot."""
    if gate is None:
        yield
        return

    gate.acquire()

    try:
        yield
    finally:
        gate.release()
