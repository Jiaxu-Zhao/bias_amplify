#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bias_dynamics.comparable_splits import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Build comparable method-specific splits from one shared source dataset")
    parser.add_argument("--protocol", required=True, help="config/comparable_training_protocol.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(protocol_path=args.protocol, output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
