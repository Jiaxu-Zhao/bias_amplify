from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Dict, List

from bias_dynamics.config import ExperimentConfig
from bias_dynamics.evaluator import bias_stability, evaluate_state
from bias_dynamics.methods import ModelState, build_method
from bias_dynamics.prompts import build_identity_conditioned_prompts


def _initial_state() -> ModelState:
    return ModelState(bias_level=0.18, perplexity=18.0, accuracy=0.62, diversity=0.9)


def run_iterative_experiment(config: ExperimentConfig, output_dir: Path) -> Dict[str, List[dict]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = build_identity_conditioned_prompts(config.min_variants_per_category)
    all_results: Dict[str, List[dict]] = {}

    for method_name in config.methods:
        method = build_method(method_name)
        runs: List[List[dict]] = []

        for run_idx in range(config.runs_per_method):
            rng = random.Random(config.seed + run_idx)
            state = _initial_state()
            run_results: List[dict] = []

            baseline = evaluate_state(state, prompts)
            bias0 = baseline.bias_score

            for iteration in range(config.iterations + 1):
                if iteration > 0:
                    state = method.update_state(state, rng)
                ev = evaluate_state(state, prompts)
                row = {
                    "method": method_name,
                    "run": run_idx,
                    "iteration": iteration,
                    "bias_score": round(ev.bias_score, 6),
                    "BAR": round(ev.bias_score / bias0, 6) if bias0 else 0.0,
                    "toxicity_gap": round(ev.toxicity_gap, 6),
                    "sentiment_gap": round(ev.sentiment_gap, 6),
                    "stereotype_score": round(ev.stereotype_score, 6),
                    "category_bias": {k: round(v, 6) for k, v in ev.category_bias.items()},
                    "perplexity": round(ev.perplexity, 6),
                    "accuracy": round(ev.accuracy, 6),
                    "diversity": round(ev.diversity, 6),
                }
                run_results.append(row)

            runs.append(run_results)

        flattened = [row for run in runs for row in run]
        all_results[method_name] = flattened
        (output_dir / f"{method_name}_iterative.json").write_text(
            json.dumps(flattened, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        bias_by_run = [run[-1]["bias_score"] for run in runs]
        summary = {
            "method": method_name,
            "final_bias_mean": sum(bias_by_run) / len(bias_by_run),
            "final_bias_variance": bias_stability(bias_by_run),
            "iterations": config.iterations,
        }
        (output_dir / f"{method_name}_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return all_results


def run_bias_injection_experiment(config: ExperimentConfig, output_dir: Path) -> List[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = build_identity_conditioned_prompts(config.min_variants_per_category)
    method = build_method("self_distill")
    rows: List[dict] = []

    for alpha in config.bias_injection.alphas:
        rng = random.Random(config.seed + int(alpha * 1000))
        state = _initial_state()
        bias0 = evaluate_state(state, prompts).bias_score

        for _ in range(config.iterations):
            state = method.update_state(state, rng, alpha=alpha)

        ev = evaluate_state(state, prompts)
        rows.append(
            {
                "method": "self_distill",
                "alpha": alpha,
                "iteration": config.iterations,
                "bias_score": round(ev.bias_score, 6),
                "BAR": round(ev.bias_score / bias0, 6) if bias0 else 0.0,
                "toxicity_gap": round(ev.toxicity_gap, 6),
                "sentiment_gap": round(ev.sentiment_gap, 6),
                "category_bias": {k: round(v, 6) for k, v in ev.category_bias.items()},
                "perplexity": round(ev.perplexity, 6),
            }
        )

    (output_dir / "self_distill_bias_injection.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return rows
