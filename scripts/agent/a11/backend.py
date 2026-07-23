from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    BACKEND_BUILD_DIR,
    BACKEND_DIR,
    PROJECT_ROOT,
    SIEVE_POST_BKZ_LOOPS,
)
from .io_utils import parse_fplll
from .scheduler import reduction_slot


# ============================================================
# Backend module loading
# ============================================================

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


# ============================================================
# Process-local persistent-sieve client
# ============================================================

_PROCESS_SIEVE_CLIENT = None


def set_process_sieve_client(client) -> None:
    """Attach one IPC sieve client to the current env process.

    This is called by workers.py before LatticeEnv is constructed.

    Keeping this registration process-local means environment.py does not need
    to know anything about multiprocessing queues or GPU worker processes.
    """
    global _PROCESS_SIEVE_CLIENT
    _PROCESS_SIEVE_CLIENT = client


def clear_process_sieve_client() -> None:
    global _PROCESS_SIEVE_CLIENT
    _PROCESS_SIEVE_CLIENT = None


# ============================================================
# Numeric helpers
# ============================================================

_LOG2 = math.log(2.0)


def _log_abs_int(value: int) -> float:
    value = abs(int(value))

    if value == 0:
        return float("-inf")

    bits = value.bit_length()
    shift = max(
        0,
        bits - 53,
    )

    mantissa = value >> shift

    return math.log(float(mantissa)) + shift * _LOG2


def _row_log_norm(row) -> float:
    logs = [_log_abs_int(value) for value in row if int(value) != 0]

    if not logs:
        return -690.0

    maximum = max(logs)

    scaled_squared_norm = sum(math.exp(2.0 * (value - maximum)) for value in logs)

    return maximum + 0.5 * math.log(
        max(
            scaled_squared_norm,
            1e-300,
        )
    )


# ============================================================
# Matrix-owner backend
# ============================================================


class LatticeBackend:
    """Exact CPU matrix owner with an IPC client for GPU sieving.

    Matrix IDs remain process-local because the C++ matrix pool is process-local.

    CPU operations:
        exact MPZ matrix
        LLL
        BKZ
        enumeration

    GPU operation:
        exact block serialization
        -> persistent GPU worker
        -> exact MPZ recovered block
        -> exact validation and whole-block write-back
    """

    def __init__(
        self,
        cpu_gate=None,
        gpu_gate=None,
        global_gpu_gate=None,
        gpu_id: int | None = None,
        sieve_client=None,
    ):
        del gpu_gate
        del global_gpu_gate

        self.cpu_gate = cpu_gate
        self.gpu_id = gpu_id

        if sieve_client is not None:
            self.sieve_client = sieve_client
        else:
            self.sieve_client = _PROCESS_SIEVE_CLIENT

    # --------------------------------------------------------
    # Module validation
    # --------------------------------------------------------

    @staticmethod
    def module_info() -> dict[str, Any]:
        cuda_available = False

        if hasattr(
            my_project_backend,
            "cuda_available",
        ):
            try:
                cuda_available = bool(my_project_backend.cuda_available())
            except Exception:
                cuda_available = False

        threshold = None

        if hasattr(
            my_project_backend,
            "adaptive_sieve_threshold",
        ):
            try:
                threshold = int(my_project_backend.adaptive_sieve_threshold())
            except Exception:
                threshold = None

        return {
            "file": getattr(
                my_project_backend,
                "__file__",
                "?",
            ),
            "has_reduce_extreme": hasattr(
                my_project_backend,
                "reduce_extreme",
            ),
            "has_reduce_bkz2_global": hasattr(
                my_project_backend,
                "reduce_bkz2_global",
            ),
            "has_extract_block": hasattr(
                my_project_backend,
                "extract_block",
            ),
            "has_apply_external_block": hasattr(
                my_project_backend,
                "apply_external_block",
            ),
            "has_sieve_reduce_serialized": hasattr(
                my_project_backend,
                "sieve_reduce_serialized",
            ),
            "has_full_lll": hasattr(
                my_project_backend,
                "full_lll",
            ),
            "has_action_uses_gpu": hasattr(
                my_project_backend,
                "action_uses_gpu",
            ),
            "adaptive_sieve_threshold": threshold,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_available": cuda_available,
        }

    @staticmethod
    def validate_required_api() -> None:
        module_file = Path(
            getattr(
                my_project_backend,
                "__file__",
                "",
            )
        ).resolve()

        expected_build_dir = Path(BACKEND_BUILD_DIR).resolve()

        if expected_build_dir not in module_file.parents:
            raise RuntimeError(
                "A11 loaded my_project_backend from the wrong location:\n"
                f"  loaded   = {module_file}\n"
                f"  expected = {expected_build_dir}"
            )

        required = (
            "create_matrix_lll",
            "clone_matrix",
            "free_matrix",
            "dump_matrix",
            "evaluate_matrix",
            "reduce_extreme",
            "reduce_bkz2_global",
            "extract_block",
            "apply_external_block",
            "sieve_reduce_serialized",
            "full_lll",
            "action_uses_gpu",
            "adaptive_sieve_threshold",
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
                "Backend is missing persistent-sieve APIs:\n  " + "\n  ".join(missing)
            )

    # --------------------------------------------------------
    # Matrix ownership
    # --------------------------------------------------------

    def create_matrix_lll(
        self,
        matrix_str: str,
    ) -> int:
        with reduction_slot(self.cpu_gate):
            return int(my_project_backend.create_matrix_lll(matrix_str))

    def clone_matrix(
        self,
        matrix_id: int,
    ) -> int:
        return int(my_project_backend.clone_matrix(matrix_id))

    def free_matrix(
        self,
        matrix_id: int,
    ) -> None:
        my_project_backend.free_matrix(matrix_id)

    def dump_matrix(
        self,
        matrix_id: int,
    ) -> str:
        return str(my_project_backend.dump_matrix(matrix_id))

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    def evaluate(
        self,
        matrix_id: int,
    ) -> dict[str, Any]:
        output = dict(my_project_backend.evaluate_matrix(matrix_id))

        output["gs_log_norms"] = np.asarray(
            output["gs_log_norms"],
            dtype=np.float64,
        )

        if "cos_matrix" in output:
            output["cos_matrix"] = np.asarray(
                output["cos_matrix"],
                dtype=np.float32,
            )

        # Compatibility fallback when the native evaluator does not return
        # the row-norm product used by the reward function.
        if "log_prod" not in output:
            rows = parse_fplll(self.dump_matrix(matrix_id))

            row_logs = [_row_log_norm(row) for row in rows]

            output["log_prod"] = float(sum(row_logs))

            if row_logs:
                min_log = min(row_logs)

                if min_log < 700.0:
                    output["min_norm"] = float(math.exp(min_log))
                else:
                    output["min_norm"] = float("inf")
            else:
                output["min_norm"] = 0.0

        return output

    # --------------------------------------------------------
    # Initial global reduction
    # --------------------------------------------------------

    def initial_bkz(
        self,
        matrix_id: int,
        beta: int = 40,
    ) -> dict[str, Any]:
        with reduction_slot(self.cpu_gate):
            raw = my_project_backend.reduce_bkz2_global(
                matrix_id,
                beta,
                1,
            )

        output = dict(raw)
        output.update(self.evaluate(matrix_id))

        return output

    # --------------------------------------------------------
    # Persistent GPU sieve route
    # --------------------------------------------------------

    def _run_gpu_sieve(
        self,
        matrix_id: int,
        pos: int,
        beta: int,
    ) -> dict[str, Any]:
        if self.sieve_client is None:
            raise RuntimeError(
                "This action requires a persistent sieve worker, "
                "but no process-local SieveClient is registered."
            )

        # Matrix IDs cannot cross processes. Extract and serialize the exact
        # MPZ block owned by this environment process.
        block_matrix = str(
            my_project_backend.extract_block(
                matrix_id,
                pos,
                beta,
            )
        )

        worker_result = self.sieve_client.reduce(
            block_matrix=block_matrix,
            beta=beta,
        )

        apply_result: dict[str, Any] = {
            "accepted": False,
            "time_ms": 0.0,
        }

        exact_block = str(
            worker_result.get(
                "block_matrix",
                "",
            )
        )

        may_apply = (
            bool(
                worker_result.get(
                    "changed",
                    False,
                )
            )
            and bool(
                worker_result.get(
                    "exact_recovery",
                    False,
                )
            )
            and bool(exact_block)
        )

        if may_apply:
            # The exact CPU-side matrix owner validates and writes the whole
            # block. The GPU worker never directly mutates this matrix pool.
            with reduction_slot(self.cpu_gate):
                apply_result = dict(
                    my_project_backend.apply_external_block(
                        matrix_id,
                        pos,
                        exact_block,
                        SIEVE_POST_BKZ_LOOPS,
                        False,
                    )
                )

        worker_time_ms = float(
            worker_result.get(
                "time_ms",
                0.0,
            )
        )

        apply_time_ms = float(
            apply_result.get(
                "time_ms",
                0.0,
            )
        )

        output = dict(worker_result)

        output.update(
            {
                "backend": "persistent_bgj_sieve",
                "accepted": bool(
                    apply_result.get(
                        "accepted",
                        False,
                    )
                ),
                "scheduled_device": "gpu",
                "physical_gpu": (
                    int(self.gpu_id)
                    if self.gpu_id is not None
                    else worker_result.get("physical_gpu")
                ),
                "time_ms": (worker_time_ms + apply_time_ms),
                "sieve_time_ms": worker_time_ms,
                "apply_time_ms": apply_time_ms,
                "apply_pot_before": apply_result.get("pot_before"),
                "apply_pot_after": apply_result.get("pot_after"),
                "apply_b1_before": apply_result.get("b1_before"),
                "apply_b1_after": apply_result.get("b1_after"),
            }
        )

        return output

    # --------------------------------------------------------
    # Agent action
    # --------------------------------------------------------

    def reduce(
        self,
        matrix_id: int,
        pos: int,
        beta: int,
    ) -> dict[str, Any]:
        uses_gpu = bool(
            my_project_backend.action_uses_gpu(
                matrix_id,
                pos,
                beta,
            )
        )

        if uses_gpu:
            output = self._run_gpu_sieve(
                matrix_id=matrix_id,
                pos=pos,
                beta=beta,
            )
        else:
            with reduction_slot(self.cpu_gate):
                raw = my_project_backend.reduce_extreme(
                    matrix_id,
                    pos,
                    beta,
                    True,
                )

            output = dict(raw)

            output.update(
                {
                    "scheduled_device": "cpu",
                    "physical_gpu": self.gpu_id,
                }
            )

        output.update(self.evaluate(matrix_id))

        return output

    # --------------------------------------------------------
    # Episode-end polish
    # --------------------------------------------------------

    def final_polish(
        self,
        matrix_id: int,
        beta: int = 45,
    ) -> dict[str, Any]:
        output = self._run_gpu_sieve(
            matrix_id=matrix_id,
            pos=0,
            beta=beta,
        )

        # The final whole-basis LLL is always exact and remains in the
        # matrix-owning environment process.
        with reduction_slot(self.cpu_gate):
            my_project_backend.full_lll(matrix_id)

        output["backend"] = "persistent_bgj_sieve_final"

        output.update(self.evaluate(matrix_id))

        return output
