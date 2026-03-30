import argparse

from bias_dynamics.config import ExperimentConfig
from bias_dynamics.main import _apply_overrides


def test_apply_overrides():
    cfg = ExperimentConfig.from_yaml("config/experiment_config.yaml")
    args = argparse.Namespace(
        seed=7,
        runs_per_method=1,
        iterations=2,
        min_variants_per_category=10,
        methods="self_distill,dpo",
    )
    new_cfg = _apply_overrides(cfg, args)

    assert new_cfg.seed == 7
    assert new_cfg.runs_per_method == 1
    assert new_cfg.iterations == 2
    assert new_cfg.min_variants_per_category == 10
    assert new_cfg.methods == ["self_distill", "dpo"]
