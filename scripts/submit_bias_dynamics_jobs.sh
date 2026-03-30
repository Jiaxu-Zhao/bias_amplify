#!/bin/bash
# submit_bias_dynamics_jobs.sh
# Usage: bash scripts/submit_bias_dynamics_jobs.sh
# Submit one sbatch job per (experiment_type x method x seed)

set -euo pipefail

# =======================
# EXPERIMENT CONFIG
# =======================
EXPERIMENT_TYPES=(
  "iterative"
  "bias_injection"
)

METHODS=(
  "sft"
  "rlhf"
  "dpo"
  "self_distill"
)

SEEDS=(42 43 44)
ITERATIONS=5
RUNS_PER_METHOD=1
MIN_VARIANTS=50

# =======================
# SLURM CONFIG
# =======================
PARTITION="normal"
ACCOUNT="aa010"
GPUS=1
CPUS=8
MEM="32G"
TIME="04:00:00"

# =======================
# PATHS / ENV
# =======================
PROJECT_DIR="$HOME/Projects/bias_amplify"
CONFIG_PATH="$PROJECT_DIR/config/experiment_config.yaml"
OUTPUT_ROOT="$PROJECT_DIR/outputs/slurm"
LOG_DIR="$PROJECT_DIR/logs"
CONDA_ENV="biasdynamics"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g'
}

total=0

for exp_type in "${EXPERIMENT_TYPES[@]}"; do
  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      SAFE_EXP=$(slugify "$exp_type")
      SAFE_METHOD=$(slugify "$method")
      JOB_NAME="biasdyn_${SAFE_EXP}_${SAFE_METHOD}_s${seed}"
      OUT_DIR="$OUTPUT_ROOT/${SAFE_EXP}/${SAFE_METHOD}/seed_${seed}"

      if [[ "$exp_type" == "iterative" ]]; then
        CMD="python -u -m bias_dynamics.main --seed ${seed} --iterations ${ITERATIONS} --runs-per-method ${RUNS_PER_METHOD} --min-variants-per-category ${MIN_VARIANTS} --methods ${method} run-all --config ${CONFIG_PATH} --output-dir ${OUT_DIR}"
      else
        # bias injection only applies to self_distill
        if [[ "$method" != "self_distill" ]]; then
          continue
        fi
        CMD="python -u -m bias_dynamics.main --seed ${seed} --iterations ${ITERATIONS} --runs-per-method ${RUNS_PER_METHOD} --min-variants-per-category ${MIN_VARIANTS} --methods self_distill bias-injection --config ${CONFIG_PATH} --output-dir ${OUT_DIR}"
      fi

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
echo "HOST: \\$(hostname)"
echo "CUDA_VISIBLE_DEVICES=\\${CUDA_VISIBLE_DEVICES:-}"

eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

cd ${PROJECT_DIR}
mkdir -p ${OUT_DIR}

${CMD}

status=\\$?
echo "EXIT_CODE: \\$status"
echo "DONE: \\$(date)"
exit \\$status
SBATCH_EOF

      echo "Submitted: ${JOB_NAME}"
      total=$((total + 1))
    done
  done
done

echo ""
echo "Total jobs submitted: ${total}"
echo "Monitor with: squeue -u \$USER"
