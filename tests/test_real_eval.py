from pathlib import Path

from bias_dynamics.real_eval import EvalRunConfig, _build_lm_eval_cmd, extract_metrics


def test_build_lm_eval_cmd():
    cfg = EvalRunConfig(
        model_name_or_path="/tmp/model",
        output_path=Path("/tmp/out.json"),
        device="cpu",
        batch_size="4",
        limit=10,
        tasks=["bbq", "mmlu"],
    )
    cmd = _build_lm_eval_cmd(cfg)
    assert "lm_eval" in cmd[0]
    assert "--tasks" in cmd
    assert "bbq,mmlu" in cmd
    assert "--limit" in cmd


def test_extract_metrics():
    fake = {
        "results": {
            "bbq": {"acc,none": 0.62},
            "crows_pairs": {"acc,none": 0.55},
            "mmlu": {"acc,none": 0.44},
            "hellaswag": {"acc_norm,none": 0.71},
        }
    }
    out = extract_metrics(fake, bias_tasks=["bbq", "crows_pairs"], capability_tasks=["mmlu", "hellaswag"])
    assert abs(out["bias_mean"] - ((0.62 + 0.55) / 2)) < 1e-9
    assert abs(out["capability_mean"] - ((0.44 + 0.71) / 2)) < 1e-9
