"""Anthropic judge via a forced strict tool call.

Anthropic has no dedicated structured-output parameter, so schema adherence is
obtained by forcing a single `strict: true` tool. Included as a third Gate 0
candidate and as a cross-family judge for the self-preference check -- rubric
scoring does not dissolve self-preference bias, so a same-family judge cannot
measure it.

Note: `temperature` is not sent. It is removed on current frontier Claude models
and returns HTTP 400; a judge that 400s emits nothing, silently. Keeping it
unset is correct on every model here.

UNVERIFIED against a live key in this repo.
"""

from __future__ import annotations

import time
from typing import Any

from ..budget import Budget
from ..criteria import Criterion
from ..prompt import JudgePrompt
from ..schema import group_schema, parse_evaluations
from .base import JudgeResult, ProviderError

_TOOL_NAME = "record_evaluation"


class AnthropicJudge:
    def __init__(
        self,
        model: str,
        price_in_per_m: float,
        price_out_per_m: float,
        *,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "anthropic not installed; `uv sync --extra anthropic`"
            ) from exc

        self.name = "anthropic"
        self.model = model
        self.price_in_per_m = price_in_per_m
        self.price_out_per_m = price_out_per_m
        self._client = anthropic.Anthropic(timeout=timeout, max_retries=max_retries)

    def judge(
        self, prompt: JudgePrompt, group: tuple[Criterion, ...], budget: Budget
    ) -> JudgeResult:
        tool: dict[str, Any] = {
            "name": _TOOL_NAME,
            "description": "Record the rubric evaluation.",
            "input_schema": group_schema(group),
            "strict": True,
        }
        started = time.perf_counter()
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=budget.max_output_tokens,
                system=prompt.system,
                messages=[{"role": "user", "content": prompt.user}],
                tools=[tool],
                tool_choice={"type": "tool", "name": _TOOL_NAME},
            )
        except Exception as exc:
            raise ProviderError(f"{self.model}: request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        if message.stop_reason == "max_tokens":
            raise ProviderError(
                f"{self.model}: truncated at max_tokens={budget.max_output_tokens} "
                f"for group_size={budget.group_size}"
            )

        payload = None
        for block in message.content:
            if getattr(block, "type", None) == "tool_use" and block.name == _TOOL_NAME:
                payload = block.input
                break
        if payload is None:
            raise ProviderError(f"{self.model}: no tool_use block ({message.stop_reason})")

        usage = message.usage
        return JudgeResult(
            evaluations=parse_evaluations(payload, group),
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            latency_ms=latency_ms,
        )
