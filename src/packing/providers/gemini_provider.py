"""Gemini judge via response JSON schema.

Included for the Gate 0 model-selection pilot: `gemini-2.5-flash-lite` is the
cheapest credible candidate, and the study needs the *cheapest model with
headroom*, not the cheapest model.

UNVERIFIED against a live key in this repo. Gemini's schema dialect is narrower
than OpenAI's, so the schema is passed through `_gemini_schema` to drop keywords
it rejects.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..budget import Budget
from ..criteria import Criterion
from ..prompt import JudgePrompt
from ..schema import group_schema, parse_evaluations
from .base import JudgeResult, ProviderError

# Gemini's responseSchema accepts a restricted subset; `additionalProperties`
# in particular is rejected. Strictness is recovered by `parse_evaluations`,
# which raises on any criterion mismatch.
_DROP = {"additionalProperties", "$schema", "title"}


def _gemini_schema(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _gemini_schema(v) for k, v in node.items() if k not in _DROP}
    if isinstance(node, list):
        return [_gemini_schema(v) for v in node]
    return node


class GeminiJudge:
    def __init__(
        self,
        model: str,
        price_in_per_m: float,
        price_out_per_m: float,
        *,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "google-genai not installed; `uv sync --extra gemini`"
            ) from exc

        self.name = "gemini"
        self.model = model
        self.price_in_per_m = price_in_per_m
        self.price_out_per_m = price_out_per_m
        self._genai = genai
        self._client = genai.Client(
            http_options=genai.types.HttpOptions(
                timeout=round(timeout * 1000),
                retry_options=genai.types.HttpRetryOptions(
                    attempts=max_retries + 1
                ),
            )
        )

    def judge(
        self, prompt: JudgePrompt, group: tuple[Criterion, ...], budget: Budget
    ) -> JudgeResult:
        started = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt.user,
                config={
                    "system_instruction": prompt.system,
                    "response_mime_type": "application/json",
                    "response_schema": _gemini_schema(group_schema(group)),
                    "max_output_tokens": budget.max_output_tokens,
                },
            )
        except Exception as exc:
            raise ProviderError(f"{self.model}: request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        text = getattr(response, "text", None)
        if not text:
            raise ProviderError(f"{self.model}: empty response")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.model}: non-JSON content: {exc}") from exc

        try:
            evaluations = parse_evaluations(payload, group)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ProviderError(
                f"{self.model}: malformed model output: {exc}"
            ) from exc

        usage = getattr(response, "usage_metadata", None)
        return JudgeResult(
            evaluations=evaluations,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            reasoning_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
            cached_input_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
            latency_ms=latency_ms,
        )
