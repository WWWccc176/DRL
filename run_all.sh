#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# 1) make sure the backend is built and importable
python -c "from backend_loader import import_backend; import_backend(); print('backend OK')"

# 2) run benchmark (80% CPU, dims 70..110). Tune betas/seeds as needed.
python bench.py --dims 70-110 --seeds-per-dim 3 --steps-mult 4 \
  --beta-bkz 20 --beta-enum 30 --beta-g6k 40
# 3) build tables + heatmap + scatter
python analyse.py
