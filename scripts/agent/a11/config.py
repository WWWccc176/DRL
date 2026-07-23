from __future__ import annotations

import os
from pathlib import Path


AGENT_VERSION = "a11"

_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROJECT_ROOT = os.environ.get(
    "DRL_ROOT",
    str(_DEFAULT_PROJECT_ROOT),
)

BACKEND_DIR = os.path.join(
    PROJECT_ROOT,
    "Backend",
)

BACKEND_BUILD_DIR = os.path.join(
    BACKEND_DIR,
    "build",
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    AGENT_VERSION,
)

CHECKPOINT_FILE = f"{AGENT_VERSION}.pth"

SEED = 42


# ============================================================
# Network
# ============================================================

FEAT_C = 64
GS_EMB = 16
CTX_DIM = 64
ACT_EMB = 5
GS_LOC = 3

DILATIONS = [1, 2, 4, 8]

BETA_REF = 60.0
DIM_REF = 60.0

NUM_GLOBALS = 7


# ============================================================
# Environment
# ============================================================

DIM_MIN = 50
DIM_MAX = 95

ENV_COUNT = 48
ENVS_PER_FILE = 2

STATE_PHASE_PERIOD = 3

INITIAL_BKZ_BETA = 40
FINAL_POLISH_BETA = 45

DETAIL_EVERY_CYCLES = 10
ACTION_BETA_RATIO = 0.8


# ============================================================
# CPU scheduler
# ============================================================

EXPECTED_LOGICAL_CPUS = int(
    os.environ.get(
        "A11_EXPECTED_LOGICAL_CPUS",
        "96",
    )
)

MAIN_CPU_THREADS = int(
    os.environ.get(
        "A11_MAIN_CPU_THREADS",
        "4",
    )
)

ENV_CPU_THREADS = int(
    os.environ.get(
        "A11_ENV_CPU_THREADS",
        "2",
    )
)

# Only CPU BKZ/enumeration jobs are controlled by this semaphore.
# GPU sieving is handled by four independent persistent workers.
CPU_REDUCTION_CONCURRENCY = int(
    os.environ.get(
        "A11_CPU_CONCURRENCY",
        "28",
    )
)


# ============================================================
# Persistent GPU sieve workers
# ============================================================

GPU_IDS = tuple(
    int(item.strip())
    for item in os.environ.get(
        "A11_GPU_IDS",
        "0,1,2,3",
    ).split(",")
    if item.strip()
)

if not GPU_IDS:
    raise ValueError("A11_GPU_IDS must contain at least one physical GPU id")

# There is exactly one persistent process per physical GPU.
PERSISTENT_SIEVE_WORKERS = len(GPU_IDS)

# Kept for compatibility with the current main.py logging.
# This no longer means a global semaphore.
GLOBAL_GPU_SIEVE_CONCURRENCY = PERSISTENT_SIEVE_WORKERS
GPU_REDUCTIONS_PER_DEVICE = 1

# The old MemAvailable gate is disabled. Each sieve task receives
# an explicit native memory budget instead.
GPU_SIEVE_MIN_AVAILABLE_GB = 0.0
GPU_SIEVE_MEMORY_POLL_SECONDS = 0.0

# Native adaptive routing threshold.
# The action grid uses 3, 7, 11, ..., so the first ordinary action
# above this threshold is normally beta=43.
SIEVE_MIN_BETA = int(
    os.environ.get(
        "A11_SIEVE_MIN_BETA",
        "40",
    )
)


# ============================================================
# Budgeted sieve action
# ============================================================

# Stop after collecting this many exact recovered candidates.
SIEVE_MAX_CANDIDATES = int(
    os.environ.get(
        "A11_SIEVE_MAX_CANDIDATES",
        "4",
    )
)

# Maximum number of complete BGJ stages for one RL action.
SIEVE_MAX_ROUNDS = int(
    os.environ.get(
        "A11_SIEVE_MAX_ROUNDS",
        "6",
    )
)

# The first native implementation will expose this as a
# stage-level work proxy. Zero disables it.
SIEVE_MAX_PAIRS = int(
    os.environ.get(
        "A11_SIEVE_MAX_PAIRS",
        "0",
    )
)

# Cooperative wall-clock budget, checked at BGJ stage boundaries.
SIEVE_TIME_BUDGET_S = float(
    os.environ.get(
        "A11_SIEVE_TIME_BUDGET_S",
        "180.0",
    )
)

# Explicit host working-set budget for each persistent GPU worker.
#
# Four workers × 32 GiB = 128 GiB maximum configured sieve budget.
# At dimensions <=95, the expected main vector DB is far below this.
SIEVE_MEMORY_BUDGET_MB = int(
    os.environ.get(
        "A11_SIEVE_MEMORY_BUDGET_MB",
        "32768",
    )
)

# Early return thresholds.
#
# b1 gain is measured as a relative norm improvement:
#     1 - exp(log_b1_after - log_b1_before)
#
# logPot gain is:
#     logPot_before - logPot_after
SIEVE_B1_REL_IMPROVEMENT = float(
    os.environ.get(
        "A11_SIEVE_B1_REL_IMPROVEMENT",
        "0.001",
    )
)

SIEVE_LOGPOT_IMPROVEMENT = float(
    os.environ.get(
        "A11_SIEVE_LOGPOT_IMPROVEMENT",
        "0.01",
    )
)

# Postprocessing applied by the exact matrix-owning env process
# after receiving an exact recovered block.
SIEVE_POST_BKZ_LOOPS = int(
    os.environ.get(
        "A11_SIEVE_POST_BKZ_LOOPS",
        "4",
    )
)


# ============================================================
# Dimension for free
# ============================================================

# -1 means automatic native selection.
SIEVE_FREE_DIM = int(
    os.environ.get(
        "A11_SIEVE_FREE_DIM",
        "-1",
    )
)

# Conservative cap for dimensions 50..95.
SIEVE_FREE_DIM_CAP = int(
    os.environ.get(
        "A11_SIEVE_FREE_DIM_CAP",
        "6",
    )
)


# ============================================================
# Sieve storage
# ============================================================

SIEVE_WORKDIR = os.environ.get(
    "A11_SIEVE_WORKDIR",
    os.path.join(
        RESULTS_DIR,
        "sieve_cache",
    ),
)

SIEVE_KEEP_WORKDIR = int(
    os.environ.get(
        "A11_SIEVE_KEEP_WORKDIR",
        "0",
    )
)

SIEVE_QUEUE_SIZE = int(
    os.environ.get(
        "A11_SIEVE_QUEUE_SIZE",
        str(max(16, ENV_COUNT)),
    )
)

SIEVE_RESPONSE_POLL_SECONDS = float(
    os.environ.get(
        "A11_SIEVE_RESPONSE_POLL_SECONDS",
        "5.0",
    )
)

SIEVE_SERVICE_CLOSE_SECONDS = float(
    os.environ.get(
        "A11_SIEVE_SERVICE_CLOSE_SECONDS",
        "12.0",
    )
)


# ============================================================
# Enumeration
# ============================================================

ENUM_TIME_BUDGET_S = float(
    os.environ.get(
        "A11_ENUM_TIME_BUDGET_S",
        "60.0",
    )
)


# ============================================================
# Worker shutdown
# ============================================================

WORKER_CLOSE_GRACE_SECONDS = float(
    os.environ.get(
        "A11_WORKER_CLOSE_GRACE_SECONDS",
        "8",
    )
)

WORKER_TERMINATE_GRACE_SECONDS = float(
    os.environ.get(
        "A11_WORKER_TERMINATE_GRACE_SECONDS",
        "5",
    )
)

WORKER_KILL_GRACE_SECONDS = float(
    os.environ.get(
        "A11_WORKER_KILL_GRACE_SECONDS",
        "2",
    )
)


# ============================================================
# DDQN
# ============================================================

BATCH_SIZE = 128
DIMS_PER_UPDATE = 3
CAPACITY_PER_DIM = 12000

TOTAL_UPDATES = 200000
TRAIN_EVERY = 4
LOG_EVERY = 4000
SAVE_EVERY = 8000

GOAL_THRESHOLD = 0.85
