# Bias Dynamics in Self-Improving Language Models

该项目实现了一个可复现实验框架，用于研究不同训练范式（SFT / RLHF(PPO-like) / DPO / Self-distillation）下 **social bias**（gender / race / religion / profession）的动态变化、迭代放大和与性能的权衡。

> 说明：`self_distill` 在本项目中按 **OPSD-style**（on-policy self-distillation）建模，参考：<https://github.com/siyan-zhao/OPSD>

## 快速开始

```bash
python -m bias_dynamics.main run-all --config config/experiment_config.yaml --output-dir outputs
python -m bias_dynamics.main bias-injection --config config/experiment_config.yaml --output-dir outputs
```

## 评测训练前后模型（真实 LLM，使用标准 benchmark）

按你的要求，这里不使用自创评测，改为主流公开 benchmark：

- **Bias（至少两个）**：`BBQ`、`CrowS-Pairs`
- **Capability（至少两个）**：`MMLU`、`HellaSwag`

执行命令：

```bash
python -m bias_dynamics.real_eval \
  --before-model /path/to/base_model_or_hf_name \
  --after-model /path/to/trained_model_or_hf_name \
  --output-json outputs/eval/self_distill/before_after_eval.json \
  --bias-tasks bbq,crows_pairs \
  --capability-tasks mmlu,hellaswag
```

输出会包含：
- `before`：训练前各 benchmark 分数（bias/capability）
- `after`：训练后各 benchmark 分数
- `delta`：逐任务变化与均值变化（after - before）

> 评测后你可以直接看 `delta.bias` 与 `delta.capability` 来分析“偏见变化 vs 语言能力变化”的 tradeoff。

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

### 2) 单任务实验（srun）

```bash
bash scripts/run_bias_dynamics_srun.sh iterative self_distill 42
bash scripts/run_bias_dynamics_srun.sh bias_injection self_distill 42
```

### 3) 提交训练前后 benchmark 评测（sbatch）

```bash
bash scripts/eval_social_bias_slurm.sh /path/to/base_ckpt /path/to/after_ckpt self_distill
```

## 依赖

若要跑 benchmark 评测（`bias_dynamics.real_eval`），需安装可选依赖：

```bash
pip install -e .[eval]
```
