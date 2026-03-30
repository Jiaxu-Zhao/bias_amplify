#!/bin/bash
# submit_standard_benchmark_jobs.sh
# Run benchmark-only (no custom metrics) evaluations for common LLMs / checkpoints.

set -euo pipefail

PARTITION="normal"
ACCOUNT="aa010"
GPUS=1
CPUS=8
MEM="48G"
TIME="08:00:00"

PROJECT_DIR="$HOME/Projects/bias_amplify"
LOG_DIR="$PROJECT_DIR/logs"
PLAN_PATH="$PROJECT_DIR/config/standard_eval_plan.json"
OUT_DIR="$PROJECT_DIR/outputs/standard_benchmarks"
CONDA_ENV="biasdynamics"

mkdir -p "$LOG_DIR" "$OUT_DIR"
JOB_NAME="std_bias_cap_eval"

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

python -u -m bias_dynamics.eval_plan \
  --plan ${PLAN_PATH} \
  --output-dir ${OUT_DIR}

echo "DONE: \\$(date)"
SBATCH_EOF

echo "Submitted: ${JOB_NAME}"
