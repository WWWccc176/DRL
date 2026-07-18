#!/usr/bin/env bash
set -uo pipefail

ROOT="/home/amax/projects/DRL"
PYTHON="/home/amax/.conda/envs/drl_env/bin/python3"

cd "$ROOT"

export PATH="/home/amax/.conda/envs/drl_env/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONFAULTHANDLER=1

FPLLL_PREFIX="$(pkg-config --variable=prefix fplll 2>/dev/null || true)"

SEARCH_ROOTS=(
    "$ROOT"
    "/home/amax/workspace"
    "/home/amax/.conda/envs/drl_env"
    "/usr/local"
    "/usr/share"
)

if [[ -n "$FPLLL_PREFIX" ]]; then
    SEARCH_ROOTS=("$FPLLL_PREFIX" "${SEARCH_ROOTS[@]}")
fi

STRAT="$(
    find "${SEARCH_ROOTS[@]}" \
        -type f \
        -path '*/strategies/default.json' \
        2>/dev/null |
    head -n 1
)"

if [[ -z "$STRAT" ]]; then
    echo "[fatal] 没有找到 fplll strategies/default.json"
    echo "训练没有启动。"
    exec bash
fi

export FPLLL_STRATEGIES_JSON="$STRAT"

LOG_DIR="$ROOT/results/a10_shared/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG="$LOG_DIR/a10_${STAMP}.log"

ln -sfn "$LOG" "$ROOT/results/a10_shared/latest_train.log"

{
    echo "============================================================"
    echo "A10 training"
    echo "start time : $(date '+%F %T %Z')"
    echo "host       : $(hostname)"
    echo "python     : $PYTHON"
    echo "strategies : $FPLLL_STRATEGIES_JSON"
    echo "log        : $LOG"
    echo "============================================================"
} | tee -a "$LOG"

"$PYTHON" - <<'PY' 2>&1 | tee -a "$LOG"
import os
import my_project_backend as backend

print("backend:", os.path.realpath(backend.__file__))

info = backend.strategies_info()
print("strategies:", info)
print("cuda:", backend.cuda_available())

expected = os.path.realpath(
    "/home/amax/projects/DRL/"
    "my_project_backend.cpython-313-x86_64-linux-gnu.so"
)

loaded = os.path.realpath(backend.__file__)

if loaded != expected:
    raise SystemExit(
        f"[fatal] 加载了错误后端：{loaded}\n"
        f"期望后端：{expected}"
    )

if not info.get("from_json"):
    raise SystemExit(
        "[fatal] default.json 仍未成功加载，拒绝开始正式训练。"
    )

if int(info.get("count", 0)) < 129:
    raise SystemExit(
        f"[fatal] strategies 数量不足：{info}"
    )

print("backend preflight: OK")
PY

CHECK_RC=${PIPESTATUS[0]}

if [[ "$CHECK_RC" -ne 0 ]]; then
    echo "[fatal] 启动自检失败，训练没有开始。" | tee -a "$LOG"
    exec bash
fi

echo "[launch] 开始运行 a10.py" | tee -a "$LOG"

set +e

"$PYTHON" \
    -X faulthandler \
    -u "$ROOT/scripts/agent/a10.py" \
    2>&1 |
tee -a "$LOG"

TRAIN_RC=${PIPESTATUS[0]}

echo | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "训练进程已退出" | tee -a "$LOG"
echo "exit code : $TRAIN_RC" | tee -a "$LOG"
echo "end time  : $(date '+%F %T %Z')" | tee -a "$LOG"
echo "log       : $LOG" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

exec bash
