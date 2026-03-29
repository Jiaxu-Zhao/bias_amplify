from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Dict, List


DEFAULT_BIAS_TASKS = ["bbq", "crows_pairs"]
DEFAULT_CAPABILITY_TASKS = ["mmlu", "hellaswag"]

# common metric keys emitted by lm-eval-harness
METRIC_PRIORITY = [
    "acc_norm,none",
    "acc,none",
    "exact_match,none",
    "f1,none",
    "mc1,none",
]


@dataclass
class EvalRunConfig:
    model_name_or_path: str
    output_path: Path
    device: str = "cuda"
    batch_size: str = "auto"
    limit: int | None = None
    tasks: List[str] | None = None
    revision: str = "main"
    trust_remote_code: bool = False


def _build_model_args(model_name_or_path: str, revision: str, trust_remote_code: bool) -> str:
    args = [f"pretrained={model_name_or_path}", f"revision={revision}"]
    if trust_remote_code:
        args.append("trust_remote_code=True")
    if os.getenv("HF_TOKEN"):
        args.append(f"token={os.getenv('HF_TOKEN')}")
    return ",".join(args)


def _build_lm_eval_cmd(cfg: EvalRunConfig) -> List[str]:
    tasks = cfg.tasks or (DEFAULT_BIAS_TASKS + DEFAULT_CAPABILITY_TASKS)
    cmd = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        _build_model_args(cfg.model_name_or_path, cfg.revision, cfg.trust_remote_code),
        "--tasks",
        ",".join(tasks),
        "--device",
        cfg.device,
        "--batch_size",
        cfg.batch_size,
        "--output_path",
        str(cfg.output_path),
    ]
    if cfg.limit is not None:
        cmd.extend(["--limit", str(cfg.limit)])
    return cmd


def run_lm_eval(cfg: EvalRunConfig) -> Dict:
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_lm_eval_cmd(cfg)
    subprocess.run(cmd, check=True)

    # lm-eval may write to file or directory depending on version
    if cfg.output_path.is_file():
        return json.loads(cfg.output_path.read_text(encoding="utf-8"))

    json_files = sorted(cfg.output_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No lm-eval json found under {cfg.output_path}")
    return json.loads(json_files[-1].read_text(encoding="utf-8"))


def _pick_metric(task_result: Dict) -> float:
    for key in METRIC_PRIORITY:
        if key in task_result:
            return float(task_result[key])
    for _, value in task_result.items():
        if isinstance(value, (float, int)):
            return float(value)
    raise ValueError(f"Could not find scalar metric in task result: {task_result}")


def extract_metrics(eval_json: Dict, bias_tasks: List[str], capability_tasks: List[str]) -> Dict:
    results = eval_json.get("results", {})
    bias = {}
    capability = {}

    for task in bias_tasks:
        if task in results:
            bias[task] = _pick_metric(results[task])

    for task in capability_tasks:
        if task in results:
            capability[task] = _pick_metric(results[task])

    if not bias:
        raise ValueError(f"No bias task results found. looked for: {bias_tasks}")
    if not capability:
        raise ValueError(f"No capability task results found. looked for: {capability_tasks}")

    bias_mean = sum(bias.values()) / len(bias)
    capability_mean = sum(capability.values()) / len(capability)

    return {
        "bias": bias,
        "capability": capability,
        "bias_mean": bias_mean,
        "capability_mean": capability_mean,
    }


def compare_before_after(
    before_model: str,
    after_model: str,
    output_json: str,
    bias_tasks: List[str] | None = None,
    capability_tasks: List[str] | None = None,
    device: str = "cuda",
    batch_size: str = "auto",
    limit: int | None = None,
    revision: str = "main",
    trust_remote_code: bool = False,
) -> Dict:
    bias_tasks = bias_tasks or DEFAULT_BIAS_TASKS
    capability_tasks = capability_tasks or DEFAULT_CAPABILITY_TASKS
    all_tasks = bias_tasks + capability_tasks

    out_root = Path(output_json).parent
    before_raw_path = out_root / "before_lm_eval.json"
    after_raw_path = out_root / "after_lm_eval.json"

    before_raw = run_lm_eval(
        EvalRunConfig(
            model_name_or_path=before_model,
            output_path=before_raw_path,
            device=device,
            batch_size=batch_size,
            limit=limit,
            tasks=all_tasks,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
    )
    after_raw = run_lm_eval(
        EvalRunConfig(
            model_name_or_path=after_model,
            output_path=after_raw_path,
            device=device,
            batch_size=batch_size,
            limit=limit,
            tasks=all_tasks,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
    )

    before = extract_metrics(before_raw, bias_tasks=bias_tasks, capability_tasks=capability_tasks)
    after = extract_metrics(after_raw, bias_tasks=bias_tasks, capability_tasks=capability_tasks)

    delta_bias = {task: after["bias"][task] - before["bias"][task] for task in before["bias"]}
    delta_cap = {
        task: after["capability"][task] - before["capability"][task] for task in before["capability"]
    }

    result = {
        "before_model": before_model,
        "after_model": after_model,
        "source": "huggingface_hub_via_lm_eval_hf_backend",
        "bias_tasks": bias_tasks,
        "capability_tasks": capability_tasks,
        "before": before,
        "after": after,
        "delta": {
            "bias": delta_bias,
            "capability": delta_cap,
            "bias_mean": after["bias_mean"] - before["bias_mean"],
            "capability_mean": after["capability_mean"] - before["capability_mean"],
        },
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _parse_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark-based before/after LLM evaluation")
    parser.add_argument("--before-model", required=True, help="Hugging Face model id, e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--after-model", required=True, help="Hugging Face model id or repo path")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--bias-tasks", default=",".join(DEFAULT_BIAS_TASKS), help="e.g. bbq,crows_pairs")
    parser.add_argument(
        "--capability-tasks",
        default=",".join(DEFAULT_CAPABILITY_TASKS),
        help="e.g. mmlu,hellaswag",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="optional lm-eval sample limit for smoke runs")
    parser.add_argument("--revision", default="main", help="HF model revision/tag")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    compare_before_after(
        before_model=args.before_model,
        after_model=args.after_model,
        output_json=args.output_json,
        bias_tasks=_parse_csv(args.bias_tasks),
        capability_tasks=_parse_csv(args.capability_tasks),
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
