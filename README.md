# Bias Dynamics in Self-Improving Language Models

该项目实现了一个可复现实验框架，用于研究不同训练范式（SFT / RLHF(PPO-like) / DPO / Self-distillation）下 **social bias**（gender / race / religion / profession）的动态变化、迭代放大和与性能的权衡。

> 说明：`self_distill` 在本项目中按 **OPSD-style**（on-policy self-distillation）建模，参考：<https://github.com/siyan-zhao/OPSD>

## 快速开始

```bash
python -m bias_dynamics.main run-all --config config/experiment_config.yaml --output-dir outputs
python -m bias_dynamics.main bias-injection --config config/experiment_config.yaml --output-dir outputs
```

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

### 1) 批量提交（sbatch）

```bash
bash scripts/submit_bias_dynamics_jobs.sh
```

脚本会按 `(experiment_type × method × seed)` 提交任务，日志在 `logs/`，输出在 `outputs/slurm/`。

### 2) 交互式/单任务（srun）

```bash
# iterative
bash scripts/run_bias_dynamics_srun.sh iterative self_distill 42

# bias injection
bash scripts/run_bias_dynamics_srun.sh bias_injection self_distill 42
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

- 为了让仓库可快速运行，当前实现使用可控的“模拟训练更新器”来近似不同方法的偏置动力学行为。
- 你可以替换 `bias_dynamics/methods.py` 中的 `update_state` 逻辑，对接真实模型训练（Transformers + TRL + DPO / OPSD 实现）。
