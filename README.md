# Bias Dynamics in Self-Improving Language Models

该项目实现了一个可复现实验框架，用于研究不同训练范式（SFT / RLHF(PPO-like) / DPO / Self-distillation）下 **social bias**（gender / race / religion / profession）的动态变化、迭代放大和与性能的权衡。

> 说明：`self_distill` 在本项目中按 **OPSD-style**（on-policy self-distillation）建模，参考：<https://github.com/siyan-zhao/OPSD>

## 快速开始

```bash
python -m bias_dynamics.main run-all --config config/experiment_config.yaml --output-dir outputs
python -m bias_dynamics.main bias-injection --config config/experiment_config.yaml --output-dir outputs
```

## 评测训练前后模型（真实 LLM）

你提到要评测“训练方法前后偏见变化 + 语言模型能力变化”，本仓库已提供：

```bash
python -m bias_dynamics.real_eval \
  --before-model /path/to/base_model_or_hf_name \
  --after-model /path/to/trained_model_or_hf_name \
  --output-json outputs/eval/self_distill/before_after_eval.json \
  --min-variants-per-category 50
```

输出会包含：
- social bias：`sentiment_gap` / `toxicity_gap` / `stereotype_score` / `bias_score` / `category_bias`
- capability：`perplexity` / `qa_accuracy` / `diversity_distinct_1` / `diversity_distinct_2`
- 以及 `delta`（after - before）

## 命令行覆盖参数（方便 SLURM）

你可以临时覆盖配置里的关键参数（seed、iteration、method 等）：

```bash
python -m bias_dynamics.main \
  --seed 42 \
  --iterations 5 \
  --runs-per-method 1 \
  --methods self_distill \
  run-all --config config/experiment_config.yaml --output-dir outputs/self_distill_seed42
```

## 在 SLURM 上跑

### 1) 批量提交实验（sbatch）

```bash
bash scripts/submit_bias_dynamics_jobs.sh
```

脚本会按 `(experiment_type × method × seed)` 提交任务，日志在 `logs/`，输出在 `outputs/slurm/`。

### 2) 单任务实验（srun）

```bash
# iterative
bash scripts/run_bias_dynamics_srun.sh iterative self_distill 42

# bias injection
bash scripts/run_bias_dynamics_srun.sh bias_injection self_distill 42
```

### 3) 提交训练前后评测任务（sbatch）

```bash
bash scripts/eval_social_bias_slurm.sh /path/to/base_ckpt /path/to/after_ckpt self_distill
```

## 输出

每轮训练都会输出统一 JSON 结构（包含 category-level social bias breakdown）：

```json
{
  "method": "self_distill",
  "iteration": 3,
  "bias_score": 0.42,
  "BAR": 1.8,
  "toxicity_gap": 0.12,
  "sentiment_gap": 0.09,
  "category_bias": {
    "gender": 0.40,
    "race": 0.43,
    "religion": 0.46,
    "profession": 0.39
  },
  "perplexity": 15.2
}
```

## 说明

- 现有 `bias_dynamics.main` 仍是“模拟训练动力学”实验，用于快速迭代对比方法。
- 新增 `bias_dynamics.real_eval` 用于真实 checkpoint 的“训练前后”偏见与能力评测。
