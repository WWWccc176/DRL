from __future__ import annotations

import my_project_backend
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import BACKEND_BUILD_DIR, BACKEND_DIR, PROJECT_ROOT
from .io_utils import parse_fplll
from .scheduler import gpu_reduction_slot, reduction_slot

# Repeated insert(0) reverses priority, so insert low -> high priority.
for path in (PROJECT_ROOT, BACKEND_DIR, BACKEND_BUILD_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


_LOG2 = math.log(2.0)


def _log_abs_int(value: int) -> float:
    value = abs(int(value))
    if value == 0:
        return float("-inf")
    bits = value.bit_length()
    shift = max(0, bits - 53)
    mantissa = value >> shift
    return math.log(float(mantissa)) + shift * _LOG2


def _row_log_norm(row) -> float:
    logs = [_log_abs_int(x) for x in row if int(x) != 0]
    if not logs:
        return -690.0
    m = max(logs)
    scaled_sq = sum(math.exp(2.0 * (x - m)) for x in logs)
    return m + 0.5 * math.log(max(scaled_sq, 1e-300))


class LatticeBackend:
    """Python boundary to the local C++/CUDA lattice backend.

    The agent still chooses only (pos, beta). The native backend exposes a routing
    query so Python can acquire the correct CPU or per-GPU semaphore without making
    the algorithm choice itself.
    """

    def __init__(
        self,
        cpu_gate=None,
        gpu_gate=None,
        global_gpu_gate=None,
        gpu_id: int | None = None,
    ):
        self.cpu_gate = cpu_gate
        self.gpu_gate = gpu_gate
        self.global_gpu_gate = global_gpu_gate
        self.gpu_id = gpu_id

    @staticmethod
    def module_info() -> dict[str, Any]:
        return {
            "file": getattr(my_project_backend, "__file__", "?"),
            "has_reduce_adaptive": hasattr(my_project_backend, "reduce_adaptive"),
            "has_reduce_extreme": hasattr(my_project_backend, "reduce_extreme"),
            "has_reduce_bkz2_global": hasattr(my_project_backend, "reduce_bkz2_global"),
            "has_reduce_sieve_block": hasattr(my_project_backend, "reduce_sieve_block"),
            "has_action_uses_gpu": hasattr(my_project_backend, "action_uses_gpu"),
            "adaptive_sieve_threshold": (
                int(my_project_backend.adaptive_sieve_threshold())
                if hasattr(my_project_backend, "adaptive_sieve_threshold")
                else None
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_available": bool(
                my_project_backend.cuda_available()
                if hasattr(my_project_backend, "cuda_available")
                else False
            ),
        }

    @staticmethod
    def validate_required_api() -> None:
        module_file = Path(getattr(my_project_backend, "__file__", "")).resolve()
        build_dir = Path(BACKEND_BUILD_DIR).resolve()

        if build_dir not in module_file.parents:
            raise RuntimeError(
                "A11 loaded my_project_backend from the wrong location: "
                f"{module_file}. Expected the rebuilt module under {build_dir}."
            )

        required = (
            "reduce_bkz2_global",
            "reduce_sieve_block",
            "action_uses_gpu",
            "adaptive_sieve_threshold",
        )
        missing = [name for name in required if not hasattr(my_project_backend, name)]
        if missing:
            raise RuntimeError(
                "Backend is missing required A11 APIs: "
                + ", ".join(missing)
                + ". Replace lattice_backend.cpp with the supplied file and rebuild Backend."
            )

    def create_matrix_lll(self, matrix_str: str) -> int:
        # Initial LLL is CPU work and participates in the global CPU budget.
        with reduction_slot(self.cpu_gate):
            return int(my_project_backend.create_matrix_lll(matrix_str))

    def clone_matrix(self, matrix_id: int) -> int:
        return int(my_project_backend.clone_matrix(matrix_id))

    def free_matrix(self, matrix_id: int) -> None:
        my_project_backend.free_matrix(matrix_id)

    def dump_matrix(self, matrix_id: int) -> str:
        return str(my_project_backend.dump_matrix(matrix_id))

    def evaluate(self, matrix_id: int) -> dict[str, Any]:
        out = dict(my_project_backend.evaluate_matrix(matrix_id))
        out["gs_log_norms"] = np.asarray(out["gs_log_norms"], dtype=np.float64)
        if "cos_matrix" in out:
            out["cos_matrix"] = np.asarray(out["cos_matrix"], dtype=np.float32)

        if "log_prod" not in out:
            rows = parse_fplll(self.dump_matrix(matrix_id))
            row_logs = [_row_log_norm(row) for row in rows]
            out["log_prod"] = float(sum(row_logs))
            out["min_norm"] = float(math.exp(min(row_logs))) if row_logs else 0.0
        return out

    def initial_bkz(self, matrix_id: int, beta: int = 40) -> dict[str, Any]:
        """Run one global BKZ 2.0 tour after create_matrix_lll()."""
        with reduction_slot(self.cpu_gate):
            raw = my_project_backend.reduce_bkz2_global(matrix_id, beta, 1)
        out = dict(raw)
        out.update(self.evaluate(matrix_id))
        return out

    def reduce(self, matrix_id: int, pos: int, beta: int) -> dict[str, Any]:
        """Normal RL action; native routing decides CPU enumeration vs GPU BGJ."""
        uses_gpu = bool(my_project_backend.action_uses_gpu(matrix_id, pos, beta))
        admission_mem_gb = None

        if uses_gpu:
            with gpu_reduction_slot(
                self.gpu_gate, self.global_gpu_gate
            ) as available_gb:
                admission_mem_gb = available_gb
                if hasattr(my_project_backend, "reduce_adaptive"):
                    raw = my_project_backend.reduce_adaptive(matrix_id, pos, beta)
                elif hasattr(my_project_backend, "reduce_extreme"):
                    raw = my_project_backend.reduce_extreme(matrix_id, pos, beta, True)
                else:
                    raw = my_project_backend.reduce(matrix_id, "ORACLE", beta, pos)
        else:
            with reduction_slot(self.cpu_gate):
                if hasattr(my_project_backend, "reduce_adaptive"):
                    raw = my_project_backend.reduce_adaptive(matrix_id, pos, beta)
                elif hasattr(my_project_backend, "reduce_extreme"):
                    raw = my_project_backend.reduce_extreme(matrix_id, pos, beta, True)
                else:
                    raw = my_project_backend.reduce(matrix_id, "ORACLE", beta, pos)

        out = dict(raw)
        out["scheduled_device"] = "gpu" if uses_gpu else "cpu"
        if uses_gpu:
            if self.gpu_id is not None:
                out["physical_gpu"] = int(self.gpu_id)
            if admission_mem_gb is not None:
                out["sieve_mem_available_gb_at_start"] = float(admission_mem_gb)
        out.update(self.evaluate(matrix_id))
        return out

    def final_polish(self, matrix_id: int, beta: int = 45) -> dict[str, Any]:
        """Cycle tail: explicit local BGJ sieve at pos=0, then full-basis LLL."""
        with gpu_reduction_slot(self.gpu_gate, self.global_gpu_gate) as available_gb:
            raw = my_project_backend.reduce_sieve_block(matrix_id, 0, beta)
        out = dict(raw)
        out["scheduled_device"] = "gpu"
        if self.gpu_id is not None:
            out["physical_gpu"] = int(self.gpu_id)
        if available_gb is not None:
            out["sieve_mem_available_gb_at_start"] = float(available_gb)
        out.update(self.evaluate(matrix_id))
        return out
