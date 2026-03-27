#!/bin/bash
# run_bias_dynamics_srun.sh
# Usage:
#   bash scripts/run_bias_dynamics_srun.sh iterative self_distill 42
#   bash scripts/run_bias_dynamics_srun.sh bias_injection self_distill 42

set -euo pipefail

EXP_TYPE="${1:-iterative}"         # iterative | bias_injection
METHOD="${2:-self_distill}"         # sft | rlhf | dpo | self_distill
SEED="${3:-42}"

PARTITION="normal"
ACCOUNT="aa010"
GPUS=1
CPUS=8
MEM="32G"
TIME="04:00:00"

PROJECT_DIR="$HOME/Projects/bias_amplify"
CONFIG_PATH="$PROJECT_DIR/config/experiment_config.yaml"
OUT_DIR="$PROJECT_DIR/outputs/srun/${EXP_TYPE}/${METHOD}/seed_${SEED}"
CONDA_ENV="biasdynamics"

mkdir -p "$OUT_DIR"

if [[ "$EXP_TYPE" == "iterative" ]]; then
  RUN_CMD="python -u -m bias_dynamics.main --seed ${SEED} --iterations 5 --runs-per-method 1 --methods ${METHOD} run-all --config ${CONFIG_PATH} --output-dir ${OUT_DIR}"
else
  RUN_CMD="python -u -m bias_dynamics.main --seed ${SEED} --iterations 5 --runs-per-method 1 --methods self_distill bias-injection --config ${CONFIG_PATH} --output-dir ${OUT_DIR}"
fi

srun \
  -p "$PARTITION" \
  -A "$ACCOUNT" \
  --gres="gpu:${GPUS}" \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  -t "$TIME" \
  bash -lc "
    set -euo pipefail
    eval \"\$(conda shell.bash hook)\"
    conda activate ${CONDA_ENV}
    cd ${PROJECT_DIR}
    ${RUN_CMD}
  "

echo "Done. Outputs at: $OUT_DIR"
