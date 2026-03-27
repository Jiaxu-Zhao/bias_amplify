#!/bin/bash
# eval_social_bias_slurm.sh
# Usage:
# bash scripts/eval_social_bias_slurm.sh /path/to/base_ckpt /path/to/after_ckpt method_name

set -euo pipefail

BEFORE_MODEL="${1:?need before model path}"
AFTER_MODEL="${2:?need after model path}"
METHOD_NAME="${3:-self_distill}"

PARTITION="normal"
ACCOUNT="aa010"
GPUS=1
CPUS=8
MEM="48G"
TIME="08:00:00"

PROJECT_DIR="$HOME/Projects/bias_amplify"
LOG_DIR="$PROJECT_DIR/logs"
OUT_DIR="$PROJECT_DIR/outputs/eval/${METHOD_NAME}"
CONDA_ENV="biasdynamics"
mkdir -p "$LOG_DIR" "$OUT_DIR"

JOB_NAME="eval_bias_${METHOD_NAME}"

sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --partition=${PARTITION}
#SBATCH -A ${ACCOUNT}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}

echo "START: \\$(date)"
eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

python -u -m bias_dynamics.real_eval \
  --before-model "${BEFORE_MODEL}" \
  --after-model "${AFTER_MODEL}" \
  --output-json "${OUT_DIR}/before_after_eval.json" \
  --min-variants-per-category 50

echo "DONE: \\$(date)"
SBATCH_EOF

echo "Submitted ${JOB_NAME}"
