from __future__ import annotations

import argparse
from pathlib import Path

from bias_dynamics.config import ExperimentConfig
from bias_dynamics.experiment import run_bias_injection_experiment, run_iterative_experiment


def _apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if args.seed is not None:
        config.seed = args.seed
    if args.runs_per_method is not None:
        config.runs_per_method = args.runs_per_method
    if args.iterations is not None:
        config.iterations = args.iterations
    if args.min_variants_per_category is not None:
        config.min_variants_per_category = args.min_variants_per_category
    if args.methods:
        config.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Bias dynamics experiment runner")
    parser.add_argument("--seed", type=int, default=None, help="override random seed")
    parser.add_argument("--runs-per-method", type=int, default=None, help="override runs_per_method")
    parser.add_argument("--iterations", type=int, default=None, help="override number of iterations")
    parser.add_argument(
        "--min-variants-per-category",
        type=int,
        default=None,
        help="override minimum prompt variants per social category",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="comma-separated methods to run, e.g. sft,rlhf,dpo,self_distill",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_all = sub.add_parser("run-all", help="Run iterative experiments for all methods")
    p_all.add_argument("--config", required=True)
    p_all.add_argument("--output-dir", required=True)

    p_inj = sub.add_parser("bias-injection", help="Run alpha sweep for self-distillation")
    p_inj.add_argument("--config", required=True)
    p_inj.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    config = _apply_overrides(ExperimentConfig.from_yaml(getattr(args, "config")), args)
    output_dir = Path(args.output_dir)

    if args.cmd == "run-all":
        run_iterative_experiment(config, output_dir)
    elif args.cmd == "bias-injection":
        run_bias_injection_experiment(config, output_dir)


if __name__ == "__main__":
    main()
