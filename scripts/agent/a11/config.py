from __future__ import annotations

import os
from pathlib import Path

AGENT_VERSION = "a11"

# Project layout:
# DRL/
# ├── Backend/
# ├── dataset/
# ├── results/
# └── scripts/agent/a11/
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = os.environ.get("DRL_ROOT", str(_DEFAULT_PROJECT_ROOT))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "Backend")
BACKEND_BUILD_DIR = os.path.join(BACKEND_DIR, "build")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", AGENT_VERSION)
CHECKPOINT_FILE = f"{AGENT_VERSION}.pth"

# Reproducibility
SEED = 42

# Network
FEAT_C = 64
GS_EMB = 16
CTX_DIM = 64
ACT_EMB = 5
GS_LOC = 3
DILATIONS = [1, 2, 4, 8]
BETA_REF = 60.0
DIM_REF = 60.0
NUM_GLOBALS = 7

# Environment / dataset scheduling
DIM_MIN = 50
DIM_MAX = 95
ENV_COUNT = 48
ENVS_PER_FILE = 2
STATE_PHASE_PERIOD = 3
INITIAL_BKZ_BETA = 40
FINAL_POLISH_BETA = 45
DETAIL_EVERY_CYCLES = 10
ACTION_BETA_RATIO = 0.8

# Hardware scheduling: 2 x Xeon 6530 (48 physical cores / 96 logical threads)
# and 4 x RTX 4090. 48 fixed env slots are preserved.
EXPECTED_LOGICAL_CPUS = int(os.environ.get("A11_EXPECTED_LOGICAL_CPUS", "96"))
MAIN_CPU_THREADS = int(os.environ.get("A11_MAIN_CPU_THREADS", "4"))
ENV_CPU_THREADS = int(os.environ.get("A11_ENV_CPU_THREADS", "2"))

# With 44 CPU reductions x 2 threads + 4 main/learner CPU threads, the scheduler
# budget is 92 runnable CPU threads, about 95.8% of 96 logical threads. GPU jobs
# use separate per-device gates and may also consume light host-side CPU time.
CPU_REDUCTION_CONCURRENCY = int(
    os.environ.get("A11_CPU_CONCURRENCY", "44")
)

GPU_IDS = tuple(
    int(x.strip())
    for x in os.environ.get("A11_GPU_IDS", "0,1,2,3").split(",")
    if x.strip()
)
if not GPU_IDS:
    raise ValueError("A11_GPU_IDS must contain at least one GPU id")

# One heavy local BGJ/DH task per physical GPU at a time. Each env process sees
# only its assigned GPU, where that physical device becomes logical cuda:0.
GPU_REDUCTIONS_PER_DEVICE = int(
    os.environ.get("A11_GPU_REDUCTIONS_PER_DEVICE", "1")
)

# Training
BATCH_SIZE = 128
DIMS_PER_UPDATE = 3
CAPACITY_PER_DIM = 12000
TOTAL_UPDATES = 200000
TRAIN_EVERY = 4
LOG_EVERY = 4000
SAVE_EVERY = 8000
GOAL_THRESHOLD = 0.85
