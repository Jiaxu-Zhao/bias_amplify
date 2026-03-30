from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from bias_dynamics.real_eval import compare_before_after


def run_eval_plan(
    plan_path: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: str = "auto",
    limit: int | None = None,
    revision: str = "main",
    trust_remote_code: bool = False,
) -> List[Dict]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    bias_tasks = plan["bias_tasks"]
    capability_tasks = plan["capability_tasks"]
    pairs = plan["pairs"]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for item in pairs:
        method = item["method"]
        family = item["model_family"]
        before_model = item["before_model"]
        after_model = item["after_model"]

        safe_method = method.replace("/", "-")
        safe_family = family.lower().replace("/", "-").replace(" ", "-")
        out_json = out_dir / f"{safe_method}_{safe_family}.json"

        result = compare_before_after(
            before_model=before_model,
            after_model=after_model,
            output_json=str(out_json),
            bias_tasks=bias_tasks,
            capability_tasks=capability_tasks,
            device=device,
            batch_size=batch_size,
            limit=limit,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        row = {
            "method": method,
            "model_family": family,
            "bias_mean_before": result["before"]["bias_mean"],
            "bias_mean_after": result["after"]["bias_mean"],
            "capability_mean_before": result["before"]["capability_mean"],
            "capability_mean_after": result["after"]["capability_mean"],
            "delta_bias_mean": result["delta"]["bias_mean"],
            "delta_capability_mean": result["delta"]["capability_mean"],
            "artifact": str(out_json),
        }
        rows.append(row)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standard benchmark eval plan for multiple model pairs")
    parser.add_argument("--plan", required=True, help="Path to plan JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    run_eval_plan(
        plan_path=args.plan,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
