from __future__ import annotations

from contextlib import contextmanager
import multiprocessing as mp
import time

from .config import (
    CPU_REDUCTION_CONCURRENCY,
    GLOBAL_GPU_SIEVE_CONCURRENCY,
    GPU_IDS,
    GPU_REDUCTIONS_PER_DEVICE,
    GPU_SIEVE_MEMORY_POLL_SECONDS,
    GPU_SIEVE_MIN_AVAILABLE_GB,
)


def create_reduction_gates():
    cpu_gate = mp.BoundedSemaphore(CPU_REDUCTION_CONCURRENCY)
    global_gpu_gate = mp.BoundedSemaphore(GLOBAL_GPU_SIEVE_CONCURRENCY)
    gpu_gates = {
        gpu_id: mp.BoundedSemaphore(GPU_REDUCTIONS_PER_DEVICE) for gpu_id in GPU_IDS
    }
    return cpu_gate, global_gpu_gate, gpu_gates


def host_mem_available_gb() -> float | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    kib = int(line.split()[1])
                    return kib / (1024.0 * 1024.0)
    except Exception:
        return None
    return None


def wait_for_sieve_memory_floor() -> float | None:
    while True:
        available_gb = host_mem_available_gb()
        if available_gb is None or available_gb >= GPU_SIEVE_MIN_AVAILABLE_GB:
            return available_gb
        time.sleep(max(0.05, GPU_SIEVE_MEMORY_POLL_SECONDS))


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


@contextmanager
def gpu_reduction_slot(gpu_gate, global_gpu_gate):
    """Admit one heavy sieve without wasting a global slot on a busy GPU.

    Acquisition order is per-GPU first, global second. If two envs target the same
    physical GPU, the second waits before consuming one of the two global sieve slots.
    """
    if gpu_gate is None and global_gpu_gate is None:
        available_gb = wait_for_sieve_memory_floor()
        yield available_gb
        return

    if gpu_gate is not None:
        gpu_gate.acquire()
    try:
        if global_gpu_gate is not None:
            global_gpu_gate.acquire()
        try:
            available_gb = wait_for_sieve_memory_floor()
            yield available_gb
        finally:
            if global_gpu_gate is not None:
                global_gpu_gate.release()
    finally:
        if gpu_gate is not None:
            gpu_gate.release()
