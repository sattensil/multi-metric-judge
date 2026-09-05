"""OpenAI judge via strict Structured Outputs.

Structured Outputs is schema-constrained decoding, which is why OpenAI is the
default here: a malformed or criterion-skipping payload is close to impossible,
so parse failures do not become a confound that correlates with group size.

Two deliberate parameter choices:

- **No ``temperature``.** Miller shows lowering temperature to reduce variance can
  triple minimum variance and inject bias; variance reduction here comes from
  replicates instead. The gpt-5 reasoning family also rejects non-default
  temperature outright.
- **``reasoning_effort`` fixed** across all conditions. It must not co-vary with
  group size or it becomes an uncontrolled mediator.

UNVERIFIED: the exact request surface (``max_completion_tokens`` vs
``max_tokens``, ``reasoning_effort`` acceptance) has not been exercised against a
live key in this repo. Run ``packing select-model --limit 2`` first; it fails loudly
rather than silently degrading.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..budget import Budget
from ..criteria import Criterion
from ..prompt import JudgePrompt
from ..schema import parse_evaluations, response_format
from .base import JudgeResult, ProviderError


class OpenAIJudge:
    def __init__(
        self,
        model: str,
        price_in_per_m: float,
        price_out_per_m: float,
        *,
        reasoning_effort: str = "low",
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("openai package not installed") from exc

        self.name = "openai"
        self.model = model
        self.price_in_per_m = price_in_per_m
        self.price_out_per_m = price_out_per_m
        self.reasoning_effort = reasoning_effort
        self._client = OpenAI(timeout=timeout, max_retries=max_retries)

    def judge(
        self, prompt: JudgePrompt, group: tuple[Criterion, ...], budget: Budget
    ) -> JudgeResult:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "response_format": response_format(group),
            "max_completion_tokens": budget.max_output_tokens,
        }
        if self.model.startswith("gpt-5"):
            request["reasoning_effort"] = self.reasoning_effort

        started = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(**request)
        except Exception as exc:
            raise ProviderError(f"{self.model}: request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        choice = completion.choices[0]
        if choice.finish_reason == "length":
            # Must not be silently tolerated: a truncated payload at large group
            # sizes is exactly the token-ceiling artifact the budget control
            # exists to prevent.
            raise ProviderError(
                f"{self.model}: truncated at max_completion_tokens="
                f"{budget.max_output_tokens} for group_size={budget.group_size}"
            )

        content = choice.message.content
        if not content:
            raise ProviderError(f"{self.model}: empty content ({choice.finish_reason})")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.model}: non-JSON content: {exc}") from exc

        usage = completion.usage
        details = getattr(usage, "completion_tokens_details", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)

        return JudgeResult(
            evaluations=parse_evaluations(payload, group),
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
            cached_input_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
            latency_ms=latency_ms,
        )
