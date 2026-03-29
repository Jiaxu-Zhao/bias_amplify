import json
from pathlib import Path

import bias_dynamics.eval_plan as eval_plan


def test_run_eval_plan_writes_summary(tmp_path: Path, monkeypatch):
    plan = {
        "bias_tasks": ["bbq", "crows_pairs"],
        "capability_tasks": ["mmlu", "hellaswag"],
        "pairs": [
            {
                "method": "sft",
                "model_family": "meta-llama/Llama-3.1-8B-Instruct",
                "before_model": "a",
                "after_model": "b",
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    def fake_compare_before_after(**kwargs):
        return {
            "before": {"bias_mean": 0.5, "capability_mean": 0.4},
            "after": {"bias_mean": 0.45, "capability_mean": 0.42},
            "delta": {"bias_mean": -0.05, "capability_mean": 0.02},
        }

    monkeypatch.setattr(eval_plan, "compare_before_after", fake_compare_before_after)

    rows = eval_plan.run_eval_plan(str(plan_path), str(tmp_path / "out"), device="cpu", batch_size="1", limit=2)
    assert len(rows) == 1
    assert rows[0]["delta_bias_mean"] == -0.05
    assert (tmp_path / "out" / "summary.json").exists()
