# Bias Dynamics in Self-Improving Language Models

这个仓库现在把“训练前后评测”收敛到 **标准 benchmark**，避免自创指标。

## 核心实验原则（保证可比性）

你提的要求是对的：
- 对于 SFT / DPO / RLHF / self-distill 的对比，必须尽量使用 **同一来源数据**。
- 不同方法如果格式不同，也应由同一 source dataset 转换成不同 view，而不是换数据来源。

本仓库已提供对应协议：`config/comparable_training_protocol.json`。

## 关键回答：模型是否从 Hugging Face 加载？

**是的。** 评测通过 `lm-eval-harness` 的 `--model hf` 后端从 Hugging Face 加载模型。你传入的 `--before-model` 和 `--after-model` 都可以是 HF model id。

私有模型：

```bash
export HF_TOKEN=hf_xxx
```

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

## 可比性数据构造（最关键）

运行：

```bash
python scripts/build_comparable_splits.py \
  --protocol config/comparable_training_protocol.json \
  --output-dir data/comparable \
  --seed 42
```

每个实验会输出：
- `sft.jsonl`（prompt + chosen）
- `dpo.jsonl`（prompt + chosen/rejected）
- `rlhf_pref.jsonl`（prompt + chosen/rejected）
- `self_distill_prompts.jsonl`（prompt only）

这样四种方法共享同一 source rows（或其子集），可比性最强。

> 也就是说：你问“现有数据是否有社会偏见内容”，现在协议里已经加入 StereoSet 作为显式偏见来源。

## 评测标准（已落实，非自创）

- **Bias（至少 2 个）**：`BBQ`、`CrowS-Pairs`
- **LLM 能力（至少 2 个）**：`MMLU`、`HellaSwag`

## 单对模型评测（before vs after）

```bash
python -m bias_dynamics.real_eval \
  --before-model meta-llama/Llama-3.1-8B-Instruct \
  --after-model your-org/llama3.1-8b-sft-iter5 \
  --output-json outputs/eval/sft_llama31.json \
  --bias-tasks bbq,crows_pairs \
  --capability-tasks mmlu,hellaswag \
  --revision main
```

## 多模型/多方法批量评测

```bash
python -m bias_dynamics.eval_plan \
  --plan config/standard_eval_plan.json \
  --output-dir outputs/standard_benchmarks \
  --revision main
```

## 依赖

```bash
pip install -e .[eval]
```
