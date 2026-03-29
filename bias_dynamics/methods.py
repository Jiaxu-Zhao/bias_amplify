from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass
class ModelState:
    bias_level: float
    perplexity: float
    accuracy: float
    diversity: float


class TrainingMethod:
    name: str = "base"

    def update_state(self, state: ModelState, rng: random.Random, alpha: float | None = None) -> ModelState:
        raise NotImplementedError


class SFTMethod(TrainingMethod):
    name = "sft"

    def update_state(self, state: ModelState, rng: random.Random, alpha: float | None = None) -> ModelState:
        return ModelState(
            bias_level=max(0.0, state.bias_level * (0.98 + rng.uniform(-0.01, 0.01))),
            perplexity=max(5.0, state.perplexity - rng.uniform(0.2, 0.6)),
            accuracy=min(1.0, state.accuracy + rng.uniform(0.005, 0.02)),
            diversity=max(0.2, state.diversity - rng.uniform(0.0, 0.01)),
        )


class RLHFMethod(TrainingMethod):
    name = "rlhf"

    def update_state(self, state: ModelState, rng: random.Random, alpha: float | None = None) -> ModelState:
        return ModelState(
            bias_level=max(0.0, state.bias_level * (0.99 + rng.uniform(-0.015, 0.015))),
            perplexity=max(5.0, state.perplexity - rng.uniform(0.1, 0.45)),
            accuracy=min(1.0, state.accuracy + rng.uniform(0.003, 0.015)),
            diversity=max(0.15, state.diversity - rng.uniform(0.0, 0.015)),
        )


class DPOMethod(TrainingMethod):
    name = "dpo"

    def update_state(self, state: ModelState, rng: random.Random, alpha: float | None = None) -> ModelState:
        return ModelState(
            bias_level=max(0.0, state.bias_level * (0.95 + rng.uniform(-0.02, 0.015))),
            perplexity=max(5.0, state.perplexity - rng.uniform(0.15, 0.5)),
            accuracy=min(1.0, state.accuracy + rng.uniform(0.006, 0.022)),
            diversity=max(0.2, state.diversity - rng.uniform(0.0, 0.02)),
        )


class OPSDSelfDistillMethod(TrainingMethod):
    name = "self_distill"

    def update_state(self, state: ModelState, rng: random.Random, alpha: float | None = None) -> ModelState:
        """Approximate OPSD-style on-policy self-distillation dynamics.

        OPSD focuses on on-policy data generation + self-distillation updates; here we mimic
        its bias dynamics for controlled research experiments.

        alpha controls teacher interpolation:
          teacher = (1-alpha) * neutral + alpha * biased
        """
        injected = alpha if alpha is not None else 0.6
        amplification = 1.03 + 0.12 * injected + rng.uniform(-0.01, 0.02)
        return ModelState(
            bias_level=max(0.0, state.bias_level * amplification),
            perplexity=max(5.0, state.perplexity - rng.uniform(0.05, 0.35)),
            accuracy=min(1.0, state.accuracy + rng.uniform(0.001, 0.012)),
            diversity=max(0.1, state.diversity - rng.uniform(0.005, 0.03)),
        )


def build_method(name: str) -> TrainingMethod:
    registry = {
        "sft": SFTMethod,
        "rlhf": RLHFMethod,
        "dpo": DPOMethod,
        "self_distill": OPSDSelfDistillMethod,
    }
    if name not in registry:
        raise ValueError(f"Unknown method: {name}")
    return registry[name]()
