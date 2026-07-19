from __future__ import annotations

import os
from pathlib import Path

AGENT_VERSION = "a11"

_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = os.environ.get("DRL_ROOT", str(_DEFAULT_PROJECT_ROOT))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "Backend")
BACKEND_BUILD_DIR = os.path.join(BACKEND_DIR, "build")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", AGENT_VERSION)
CHECKPOINT_FILE = f"{AGENT_VERSION}.pth"

SEED = 42

FEAT_C = 64
GS_EMB = 16
CTX_DIM = 64
ACT_EMB = 5
GS_LOC = 3
DILATIONS = [1, 2, 4, 8]
BETA_REF = 60.0
DIM_REF = 60.0
NUM_GLOBALS = 7

DIM_MIN = 50
DIM_MAX = 95
ENV_COUNT = 48
ENVS_PER_FILE = 2
STATE_PHASE_PERIOD = 3
INITIAL_BKZ_BETA = 40
FINAL_POLISH_BETA = 45
DETAIL_EVERY_CYCLES = 10
ACTION_BETA_RATIO = 0.8

# 2 x Xeon 6530: 48 physical cores / 96 logical threads.
# Keep one env slot per physical core, but do not allow every env to enter a heavy
# native CPU reduction simultaneously. The native backend may create extra threads,
# so 28 is deliberately below the theoretical 44-job limit used previously.
EXPECTED_LOGICAL_CPUS = int(os.environ.get("A11_EXPECTED_LOGICAL_CPUS", "96"))
MAIN_CPU_THREADS = int(os.environ.get("A11_MAIN_CPU_THREADS", "4"))
ENV_CPU_THREADS = int(os.environ.get("A11_ENV_CPU_THREADS", "2"))
CPU_REDUCTION_CONCURRENCY = int(os.environ.get("A11_CPU_CONCURRENCY", "28"))

GPU_IDS = tuple(
    int(x.strip())
    for x in os.environ.get("A11_GPU_IDS", "0,1,2,3").split(",")
    if x.strip()
)
if not GPU_IDS:
    raise ValueError("A11_GPU_IDS must contain at least one GPU id")

# Each physical GPU may execute at most one heavy local BGJ/DH job.
GPU_REDUCTIONS_PER_DEVICE = int(os.environ.get("A11_GPU_REDUCTIONS_PER_DEVICE", "1"))

# Host RAM, rather than GPU count, is currently the limiting resource. The four
# per-GPU lanes remain available, but at most two heavy sieve jobs may be admitted
# globally at the same time.
GLOBAL_GPU_SIEVE_CONCURRENCY = int(
    os.environ.get("A11_GLOBAL_GPU_SIEVE_CONCURRENCY", "2")
)

# Secondary safety floor. A new sieve waits while Linux MemAvailable is below this
# value. This does not replace the global concurrency limit; it protects against
# unrelated memory pressure and retained native allocations.
GPU_SIEVE_MIN_AVAILABLE_GB = float(
    os.environ.get("A11_GPU_SIEVE_MIN_AVAILABLE_GB", "160")
)
GPU_SIEVE_MEMORY_POLL_SECONDS = float(
    os.environ.get("A11_GPU_SIEVE_MEMORY_POLL_SECONDS", "1.0")
)

# Graceful shutdown: close command first, then SIGTERM, finally SIGKILL.
WORKER_CLOSE_GRACE_SECONDS = float(
    os.environ.get("A11_WORKER_CLOSE_GRACE_SECONDS", "8")
)
WORKER_TERMINATE_GRACE_SECONDS = float(
    os.environ.get("A11_WORKER_TERMINATE_GRACE_SECONDS", "5")
)
WORKER_KILL_GRACE_SECONDS = float(os.environ.get("A11_WORKER_KILL_GRACE_SECONDS", "2"))

BATCH_SIZE = 128
DIMS_PER_UPDATE = 3
CAPACITY_PER_DIM = 12000
TOTAL_UPDATES = 200000
TRAIN_EVERY = 4
LOG_EVERY = 4000
SAVE_EVERY = 8000
GOAL_THRESHOLD = 0.85
