from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List


@dataclass
class GenerationConfig:
    temperature: float
    top_p: float
    max_length: int
    n_samples: int


@dataclass
class BiasInjectionConfig:
    alphas: List[float]
    neutral_template: str
    biased_template: str


@dataclass
class ExperimentConfig:
    seed: int
    runs_per_method: int
    iterations: int
    methods: List[str]
    generation: GenerationConfig
    min_variants_per_category: int
    bias_injection: BiasInjectionConfig

    @staticmethod
    def from_yaml(path: str | Path) -> "ExperimentConfig":
        # The config file is stored as JSON (valid YAML subset) for zero-dependency parsing.
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return ExperimentConfig(
            seed=data["seed"],
            runs_per_method=data["runs_per_method"],
            iterations=data["iterations"],
            methods=data["methods"],
            generation=GenerationConfig(**data["generation"]),
            min_variants_per_category=data["prompts"]["min_variants_per_category"],
            bias_injection=BiasInjectionConfig(**data["bias_injection"]),
        )
