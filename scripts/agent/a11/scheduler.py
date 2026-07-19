from __future__ import annotations

from contextlib import contextmanager
import multiprocessing as mp

from .config import (
    CPU_REDUCTION_CONCURRENCY,
    GPU_IDS,
    GPU_REDUCTIONS_PER_DEVICE,
)


def create_reduction_gates():
    """Create one CPU concurrency gate and one independent gate per physical GPU."""
    cpu_gate = mp.BoundedSemaphore(CPU_REDUCTION_CONCURRENCY)
    gpu_gates = {
        gpu_id: mp.BoundedSemaphore(GPU_REDUCTIONS_PER_DEVICE)
        for gpu_id in GPU_IDS
    }
    return cpu_gate, gpu_gates


@contextmanager
def reduction_slot(gate):
    if gate is None:
        yield
        return

    gate.acquire()
    try:
        yield
    finally:
        gate.release()
