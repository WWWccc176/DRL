from __future__ import annotations

import os
import random
import sys

import numpy as np

from .config import PROJECT_ROOT, SEED


def configure_main_runtime() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


def configure_env_runtime() -> None:
    # Do NOT hide CUDA here: the local C++/CUDA backend now owns sieving.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
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

    torch.set_num_threads(4)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def status(msg: str) -> None:
    sys.stdout.write("\r\033[K" + msg)
    sys.stdout.flush()


def log(msg: str) -> None:
    sys.stdout.write("\r\033[K" + msg + "\n")
    sys.stdout.flush()
