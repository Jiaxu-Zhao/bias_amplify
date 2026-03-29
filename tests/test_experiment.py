from pathlib import Path

from bias_dynamics.config import ExperimentConfig
from bias_dynamics.experiment import run_bias_injection_experiment, run_iterative_experiment


def test_end_to_end(tmp_path: Path):
    cfg = ExperimentConfig.from_yaml("config/experiment_config.yaml")
    cfg.runs_per_method = 1
    cfg.iterations = 2

    result = run_iterative_experiment(cfg, tmp_path)
    assert "self_distill" in result
    assert (tmp_path / "self_distill_iterative.json").exists()
    assert "category_bias" in result["self_distill"][0]
    assert "religion" in result["self_distill"][0]["category_bias"]

    injection = run_bias_injection_experiment(cfg, tmp_path)
    assert len(injection) == len(cfg.bias_injection.alphas)
    assert "category_bias" in injection[0]
