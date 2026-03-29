#!/usr/bin/env python3
"""Prepare moderate-size training datasets from Hugging Face for SFT / DPO / RLHF.

Usage:
  python scripts/prepare_training_data.py \
    --plan config/training_dataset_plan.json \
    --output-dir data/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Dict, List



def _sample_records(records: List[Dict], max_samples: int, seed: int) -> List[Dict]:
    if len(records) <= max_samples:
        return records
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    keep = set(idx[:max_samples])
    return [records[i] for i in range(len(records)) if i in keep]


def _to_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(plan_path: str, output_dir: str, seed: int = 42) -> Dict:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {"sft": [], "preference": []}

    for section in ["sft", "preference"]:
        for ds in plan[section]:
            name = ds["name"]
            split = ds["split"]
            max_samples = int(ds["max_samples"])

            from datasets import load_dataset

            dataset = load_dataset(name, split=split)
            rows = [dict(r) for r in dataset]
            rows = _sample_records(rows, max_samples=max_samples, seed=seed)

            file_name = f"{section}__{name.replace('/', '__')}__{split}.jsonl"
            output_file = out / file_name
            _to_jsonl(output_file, rows)

            summary[section].append(
                {
                    "dataset": name,
                    "split": split,
                    "kept": len(rows),
                    "file": str(output_file),
                }
            )

    summary["eval_bias"] = plan.get("eval_bias", [])
    summary["eval_capability"] = plan.get("eval_capability", [])

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HF training datasets for bias dynamics experiments")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare(plan_path=args.plan, output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
