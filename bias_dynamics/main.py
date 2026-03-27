from __future__ import annotations

import argparse
from pathlib import Path

from bias_dynamics.config import ExperimentConfig
from bias_dynamics.experiment import run_bias_injection_experiment, run_iterative_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Bias dynamics experiment runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_all = sub.add_parser("run-all", help="Run iterative experiments for all methods")
    p_all.add_argument("--config", required=True)
    p_all.add_argument("--output-dir", required=True)

    p_inj = sub.add_parser("bias-injection", help="Run alpha sweep for self-distillation")
    p_inj.add_argument("--config", required=True)
    p_inj.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    output_dir = Path(args.output_dir)

    if args.cmd == "run-all":
        run_iterative_experiment(config, output_dir)
    elif args.cmd == "bias-injection":
        run_bias_injection_experiment(config, output_dir)


if __name__ == "__main__":
    main()
