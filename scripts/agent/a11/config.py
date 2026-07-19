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

# Local backend scheduling.
# The integrated BGJ backend owns its internal CPU/GPU scheduling. Keep this gate
# conservative unless profiling shows that concurrent reductions are beneficial.
MAX_CONCURRENT_BACKEND_REDUCTIONS = int(
    os.environ.get("A11_BACKEND_CONCURRENCY", "1")
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
