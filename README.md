# Bias Dynamics in Self-Improving Language Models

这个仓库现在把“训练前后评测”收敛到 **标准 benchmark**，避免自创指标。

## 从零开始运行（你刚 clone 后按这个顺序）

### Step 0: 环境准备

```bash
git clone <your_repo_url>
cd bias_amplify
python -m venv .venv
source .venv/bin/activate
pip install -e .[eval]
```

> 如果你要拉取私有 Hugging Face 模型：

```bash
export HF_TOKEN=hf_xxx
```

### Step 1: 先做健康检查（确认仓库可运行）

```bash
bash scripts/check_merge_health.sh
```

### Step 2: 生成“可比性训练数据”

```bash
python scripts/build_comparable_splits.py \
  --protocol config/comparable_training_protocol.json \
  --output-dir data/comparable \
  --seed 42
```

### Step 3: 训练你的模型（SFT / DPO / RLHF / self-distill）

这个仓库目前提供了数据构造和评测管线；训练部分请使用你现有训练脚本（例如 TRL / DPOTrainer / PPO）。

要求：同一个实验组内，四种方法使用同一个 `data/comparable/<exp_name>/` 来源。

### Step 4: 评测训练前后（标准 benchmark）

```bash
python -m bias_dynamics.real_eval \
  --before-model meta-llama/Llama-3.1-8B-Instruct \
  --after-model your-org/llama3.1-8b-sft-iter5 \
  --output-json outputs/eval/sft_llama31.json \
  --bias-tasks bbq,crows_pairs \
  --capability-tasks mmlu,hellaswag \
  --revision main
```

### Step 5: 批量跑多个模型/方法

先编辑 `config/standard_eval_plan.json`，再运行：

```bash
python -m bias_dynamics.eval_plan \
  --plan config/standard_eval_plan.json \
  --output-dir outputs/standard_benchmarks \
  --revision main
```

---

## 你最需要改的参数（按优先级）

1. `config/comparable_training_protocol.json`
   - `source_dataset` / `source_split`
   - `max_samples`
   - 这是决定“训练数据来源和规模”的核心参数。

2. `config/standard_eval_plan.json`
   - `before_model` / `after_model`
   - `method` / `model_family`
   - 这是决定“评测哪些模型对”的核心参数。

3. 运行命令参数
   - `--seed`
   - `--revision`
   - `--device` / `--batch-size`

---

## 核心实验原则（保证可比性）

你提的要求是对的：
- 对于 SFT / DPO / RLHF / self-distill 的对比，必须尽量使用 **同一来源数据**。
- 不同方法如果格式不同，也应由同一 source dataset 转换成不同 view，而不是换数据来源。

本仓库已提供对应协议：`config/comparable_training_protocol.json`。

## 为什么选这些训练数据源（可写进论文方法部分）

我们推荐两组“单一来源”实验（每组内部四种训练方法共享同一来源）：

1. `HuggingFaceH4/ultrafeedback_binarized`
   - 有 prompt + chosen/rejected，天然支持 SFT/DPO/RLHF。
   - self-distill 使用同一 prompt 池，保证输入分布一致。

2. `Anthropic/hh-rlhf`
   - Helpful-Harmless 偏好对，适合对齐训练研究。
   - 同样可从同一来源构造四种训练 view。

3. `McGill-NLP/stereoset`（新增：明确包含社会刻板偏见）
   - 包含 stereotype / anti-stereotype 对，覆盖 gender / race / religion 等维度。
   - 可以直接构造：chosen=anti-stereotype, rejected=stereotype，用于 bias-sensitive 训练与放大分析。

## 评测标准（已落实，非自创）

- **Bias（至少 2 个）**：`BBQ`、`CrowS-Pairs`
- **LLM 能力（至少 2 个）**：`MMLU`、`HellaSwag`
