"""Deterministic offline judge.

Two jobs:

1. Make the whole pipeline -- planning, resume, scoring, tie accounting, analysis
   -- exercisable with no API key and no spend.
2. **Validate the estimators against a known truth.** The mock can be told to
   degrade with group size by a specified amount, so the analysis can be checked
   for its ability to recover an effect that was deliberately planted, and (just
   as important) to *not* find one when none was planted. An estimator suite that
   has never been tested against a known answer is not evidence of anything.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Callable

from ..budget import Budget
from ..criteria import Criterion
from ..prompt import JudgePrompt
from .base import JudgeResult

TruthFn = Callable[[JudgePrompt, Criterion], int]


class MockJudge:
    def __init__(
        self,
        model: str = "mock",
        price_in_per_m: float = 0.0,
        price_out_per_m: float = 0.0,
        *,
        seed: int = 0,
        base_noise: float = 0.0,
        noise_per_extra_criterion: float = 0.0,
        truth_fn: TruthFn | None = None,
    ) -> None:
        self.name = "mock"
        self.model = model
        self.price_in_per_m = price_in_per_m
        self.price_out_per_m = price_out_per_m
        self.seed = seed
        self.base_noise = base_noise
        self.noise_per_extra_criterion = noise_per_extra_criterion
        self.truth_fn = truth_fn
        self.calls = 0

    def _rng(self, prompt: JudgePrompt, group: tuple[Criterion, ...]) -> random.Random:
        key = f"{self.seed}|{prompt.user}|{'|'.join(c.key for c in group)}"
        digest = hashlib.sha256(key.encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def noise_probability(self, group_size: int) -> float:
        return min(
            1.0, self.base_noise + self.noise_per_extra_criterion * (group_size - 1)
        )

    def judge(
        self, prompt: JudgePrompt, group: tuple[Criterion, ...], budget: Budget
    ) -> JudgeResult:
        self.calls += 1
        rng = self._rng(prompt, group)
        p_noise = self.noise_probability(len(group))

        evaluations: dict[str, dict[str, Any]] = {}
        for criterion in group:
            if self.truth_fn is not None:
                score = self.truth_fn(prompt, criterion)
            else:
                score = rng.randint(criterion.scale_min, criterion.scale_max)
            if rng.random() < p_noise:
                score = rng.randint(criterion.scale_min, criterion.scale_max)
            evaluations[criterion.key] = {
                "score": int(score),
                "reasoning": f"mock reasoning for {criterion.key}",
                "evidence": "mock evidence",
            }

        return JudgeResult(
            evaluations=evaluations,
            input_tokens=len(prompt.user) // 4,
            output_tokens=60 * len(group),
            reasoning_tokens=0,
            cached_input_tokens=0,
            latency_ms=1.0,
        )
