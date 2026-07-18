from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np

from .config import BACKEND_BUILD_DIR, BACKEND_DIR, PROJECT_ROOT
from .io_utils import parse_fplll
from .scheduler import reduction_slot

for path in (BACKEND_BUILD_DIR, BACKEND_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import my_project_backend

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
    """Single Python boundary to Backend/my_project_backend.

    The RL action is only (pos, beta). Normal actions use the backend adaptive
    Enum/BGJ route. Initial BKZ40 and final sieve45+LLL use explicit native APIs so
    they cannot be confused with the adaptive route.
    """

    def __init__(self, reduction_gate=None):
        self.reduction_gate = reduction_gate

    @staticmethod
    def module_info() -> dict[str, Any]:
        return {
            "file": getattr(my_project_backend, "__file__", "?"),
            "has_reduce_adaptive": hasattr(my_project_backend, "reduce_adaptive"),
            "has_reduce_extreme": hasattr(my_project_backend, "reduce_extreme"),
            "has_reduce_bkz2_global": hasattr(my_project_backend, "reduce_bkz2_global"),
            "has_reduce_sieve_block": hasattr(my_project_backend, "reduce_sieve_block"),
            "cuda_available": bool(
                my_project_backend.cuda_available()
                if hasattr(my_project_backend, "cuda_available")
                else False
            ),
        }

    @staticmethod
    def validate_required_api() -> None:
        missing = [
            name
            for name in ("reduce_bkz2_global", "reduce_sieve_block")
            if not hasattr(my_project_backend, name)
        ]
        if missing:
            raise RuntimeError(
                "Backend is missing required A11 APIs: "
                + ", ".join(missing)
                + ". Rebuild Backend with the supplied lattice_backend.cpp."
            )

    def create_matrix_lll(self, matrix_str: str) -> int:
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
        with reduction_slot(self.reduction_gate):
            raw = my_project_backend.reduce_bkz2_global(matrix_id, beta, 1)
        out = dict(raw)
        out.update(self.evaluate(matrix_id))
        return out

    def reduce(self, matrix_id: int, pos: int, beta: int) -> dict[str, Any]:
        """Normal RL action: backend decides exact enumeration vs local BGJ sieve."""
        with reduction_slot(self.reduction_gate):
            if hasattr(my_project_backend, "reduce_adaptive"):
                raw = my_project_backend.reduce_adaptive(matrix_id, pos, beta)
            elif hasattr(my_project_backend, "reduce_extreme"):
                raw = my_project_backend.reduce_extreme(matrix_id, pos, beta, True)
            else:
                raw = my_project_backend.reduce(matrix_id, "ORACLE", beta, pos)

        out = dict(raw)
        out.update(self.evaluate(matrix_id))
        return out

    def final_polish(self, matrix_id: int, beta: int = 45) -> dict[str, Any]:
        """Cycle tail: pos=0 local sieve(beta) followed by full-basis LLL."""
        with reduction_slot(self.reduction_gate):
            raw = my_project_backend.reduce_sieve_block(matrix_id, 0, beta)
        out = dict(raw)
        out.update(self.evaluate(matrix_id))
        return out
