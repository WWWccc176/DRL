#!/usr/bin/env bash
#
# DRL lattice-reduction benchmark — full pipeline.
#
#   Stage 0  environment + sanity checks (venv, g6k root, GPU)
#   Stage 1  G6K smoke test (gate: abort everything if the sieve can't run)
#   Stage 2  CPU methods  (LLL/BKZ/ENUM) — big process pool, ~80% cores, no CUDA
#   Stage 3  G6K (GPU)    — isolated subprocess per file, single worker
#   Stage 4  analysis     — .txt + .csv tables, cosine heatmaps
#
# Usage:
#   ./run_all.sh                      # everything, dims 70-110, 3 seeds
#   ./run_all.sh --no-g6k             # CPU methods + analysis only
#   ./run_all.sh --only-g6k           # G6K + analysis only
#   ./run_all.sh --dims 70-90 --seeds 2
#   ./run_all.sh --skip-smoke         # trust the sieve, skip the gate
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Config (edit here or override via flags / env)
# ---------------------------------------------------------------------------
ANALYSE_DIR="${ANALYSE_DIR:-$HOME/DRL/scripts/analyse}"
VENV="${VENV:-$HOME/DRL/python_venv}"
G6K_ROOT="${G6K_ROOT:-$HOME/workspace/builds/g6k}"
CUDA_LIBS="${CUDA_LIBS:-/opt/cuda/lib64:/usr/local/cuda/lib64}"

DIMS="70-110"
SEEDS=0
STEPS_MULT=1
BETA_BKZ=20
BETA_ENUM=30
BETA_G6K=40
G6K_TIMEOUT=1800

RUN_CPU=1
RUN_G6K=1
RUN_SMOKE=1

# CPU pool size = 80% of logical cores (leave headroom)
NCPU="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
WORKERS_CPU="${WORKERS_CPU:-$(python3 -c "print(max(1, int(0.8*${NCPU})))")}"

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
  --no-g6k)
    RUN_G6K=0
    shift
    ;;
  --only-g6k)
    RUN_CPU=0
    shift
    ;;
  --skip-smoke)
    RUN_SMOKE=0
    shift
    ;;
  --dims)
    DIMS="$2"
    shift 2
    ;;
  --seeds)
    SEEDS="$2"
    shift 2
    ;;
  --steps-mult)
    STEPS_MULT="$2"
    shift 2
    ;;
  --beta-g6k)
    BETA_G6K="$2"
    shift 2
    ;;
  --workers)
    WORKERS_CPU="$2"
    shift 2
    ;;
  -h | --help)
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)
    echo "unknown flag: $1" >&2
    exit 2
    ;;
  esac
done

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
cd "$ANALYSE_DIR"

if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  echo "WARNING: venv not found at $VENV (continuing with system python)" >&2
fi

export G6K_ROOT
export PYTHONPATH="$G6K_ROOT:$ANALYSE_DIR:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$G6K_ROOT/kernel:$CUDA_LIBS:$VENV/lib:${LD_LIBRARY_PATH:-}"

RESULTS_DIR="$(python3 -c "from pathlib import Path;print((Path('$ANALYSE_DIR').parents[1]/'results'/'bench').resolve())")"
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

hr() { printf '%s\n' "----------------------------------------------------------------------"; }

# ---------------------------------------------------------------------------
# Stage 0 — sanity
# ---------------------------------------------------------------------------
hr
echo ">> Stage 0: environment"
echo "   analyse dir : $ANALYSE_DIR"
echo "   venv        : $VENV"
echo "   g6k root    : $G6K_ROOT"
echo "   dims        : $DIMS   seeds/dim: $([ "$SEEDS" -le 0 ] && echo ALL || echo $SEEDS)   steps_mult: $STEPS_MULT"
echo "   CPU workers : $WORKERS_CPU / $NCPU cores"
echo "   results     : $RESULTS_DIR"

python3 -c "import g6k_env; print('   g6k_env root:', g6k_env.G6K_ROOT)" ||
  {
    echo "ERROR: g6k_env import failed"
    exit 1
  }

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "   GPU:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader |
    sed 's/^/     /'
else
  echo "   GPU: nvidia-smi not found (G6K stage may fail)"
fi

# ---------------------------------------------------------------------------
# Stage 1 — G6K smoke gate
# ---------------------------------------------------------------------------
if [[ "$RUN_G6K" -eq 1 && "$RUN_SMOKE" -eq 1 ]]; then
  hr
  echo ">> Stage 1: G6K smoke test (gate)"
  if python3 g6k_smoke.py 2>&1 | tee "$LOG_DIR/smoke_${STAMP}.log" | grep -q "SMOKE OK"; then
    echo "   smoke OK — GPU sieve is functional"
  else
    echo "ERROR: G6K smoke test did NOT reach 'SMOKE OK'."
    echo "       see $LOG_DIR/smoke_${STAMP}.log"
    echo "       (run with --skip-smoke to bypass, or --no-g6k for CPU-only)"
    exit 1
  fi
else
  hr
  echo ">> Stage 1: G6K smoke test SKIPPED"
fi

# ---------------------------------------------------------------------------
# Stage 2 — CPU methods (big pool, CUDA disabled inside workers)
# ---------------------------------------------------------------------------
if [[ "$RUN_CPU" -eq 1 ]]; then
  hr
  echo ">> Stage 2: CPU methods (LLL,BKZ,ENUM)"
  LATTICE_DISABLE_CUDA=1 python3 bench.py \
    --dims "$DIMS" \
    --seeds-per-dim "$SEEDS" \
    --methods LLL,BKZ,ENUM \
    --workers "$WORKERS_CPU" \
    --steps-mult "$STEPS_MULT" \
    --beta-bkz "$BETA_BKZ" \
    --beta-enum "$BETA_ENUM" \
    2>&1 | tee "$LOG_DIR/cpu_${STAMP}.log"
else
  hr
  echo ">> Stage 2: CPU methods SKIPPED"
fi

# ---------------------------------------------------------------------------
# Stage 3 — G6K (GPU): ONE persistent worker, prefetch pipeline (keeps GPU busy)
# ---------------------------------------------------------------------------
if [[ "$RUN_G6K" -eq 1 ]]; then
  hr
  echo ">> Stage 3: G6K (GPU) — persistent worker, prefetch pipeline"
  python3 g6k_server.py \
    --dims "$DIMS" \
    --seeds-per-dim "$SEEDS" \
    --beta "$BETA_G6K" \
    --steps-mult "$STEPS_MULT" \
    --prefetch 2 \
    2>&1 | tee "$LOG_DIR/g6k_${STAMP}.log"
else
  hr
  echo ">> Stage 3: G6K SKIPPED"
fi
# ---------------------------------------------------------------------------
# Stage 4 — analysis (.txt + .csv tables, heatmaps)
# ---------------------------------------------------------------------------
hr
echo ">> Stage 4: analysis"
python3 analyse.py 2>&1 | tee "$LOG_DIR/analyse_${STAMP}.log"

hr
echo ">> DONE."
echo "   raw records : $RESULTS_DIR/raw"
echo "   cos csv     : $RESULTS_DIR/cos"
echo "   logs        : $LOG_DIR"
