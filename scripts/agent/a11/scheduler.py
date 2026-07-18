from __future__ import annotations

from contextlib import contextmanager
import multiprocessing as mp

from .config import MAX_CONCURRENT_BACKEND_REDUCTIONS


def create_reduction_gate():
    return mp.BoundedSemaphore(MAX_CONCURRENT_BACKEND_REDUCTIONS)


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
