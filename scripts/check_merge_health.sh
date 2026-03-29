#!/bin/bash
# check_merge_health.sh
# Quick repo health/merge check for this project.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/5] git status"
git status --short

echo "[2/5] run unit tests"
pytest -q

echo "[3/5] check key CLIs"
python -m bias_dynamics.main --help >/dev/null
python -m bias_dynamics.real_eval --help >/dev/null
python -m bias_dynamics.eval_plan --help >/dev/null
python scripts/build_comparable_splits.py --help >/dev/null

echo "[4/5] validate key config files"
python - <<'PY'
import json
from pathlib import Path
for fp in [
    'config/comparable_training_protocol.json',
    'config/standard_eval_plan.json',
    'config/training_dataset_plan.json',
]:
    json.loads(Path(fp).read_text(encoding='utf-8'))
print('config json parse: ok')
PY

echo "[5/5] done - merge health check passed"
