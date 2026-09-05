"""Provider protocol and registry.

Every provider returns the same ``JudgeResult`` so the runner is provider-blind
and the model factor is a one-word change.

Usage accounting note: ``output_tokens`` must be the *billed* completion tokens
including reasoning tokens where the provider bills them together, because the
budget control is defined against that ceiling. ``reasoning_tokens`` is reported
separately when the provider exposes it, so the analysis can check that reasoning
depth is not a hidden mediator of any group-size effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..budget import Budget
from ..criteria import Criterion
from ..prompt import JudgePrompt


class ProviderError(RuntimeError):
    """Raised for a failed or unusable judge call."""


@dataclass
class JudgeResult:
    evaluations: dict[str, dict[str, Any]]
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class JudgeProvider(Protocol):
    name: str
    model: str
    price_in_per_m: float
    price_out_per_m: float

    def judge(
        self, prompt: JudgePrompt, group: tuple[Criterion, ...], budget: Budget
    ) -> JudgeResult: ...


# Prices are USD per 1M tokens, from each vendor's published pricing page.
# Recorded here so `cost.py` can report spend without a network call; re-check
# before quoting them anywhere, since vendor pricing moves.
CATALOG: dict[str, dict[str, Any]] = {
    "gpt-5-mini": {"provider": "openai", "in": 0.25, "out": 2.00},
    "gpt-5-nano": {"provider": "openai", "in": 0.05, "out": 0.40},
    "gpt-4.1-nano": {"provider": "openai", "in": 0.10, "out": 0.40},
    "gpt-4o-mini": {"provider": "openai", "in": 0.15, "out": 0.60},
    "gemini-3.1-flash-lite": {"provider": "gemini", "in": 0.25, "out": 1.50},
    "gemini-2.5-flash-lite": {"provider": "gemini", "in": 0.10, "out": 0.40},
    "claude-haiku-4-5": {"provider": "anthropic", "in": 1.00, "out": 5.00},
    "mock": {"provider": "mock", "in": 0.0, "out": 0.0},
}


def get_provider(model: str, **kwargs: Any) -> JudgeProvider:
    try:
        entry = CATALOG[model]
    except KeyError:
        raise ProviderError(
            f"unknown model {model!r}; known: {sorted(CATALOG)}"
        ) from None

    kind = entry["provider"]
    if kind == "openai":
        from .openai_provider import OpenAIJudge

        return OpenAIJudge(model, entry["in"], entry["out"], **kwargs)
    if kind == "gemini":
        from .gemini_provider import GeminiJudge

        return GeminiJudge(model, entry["in"], entry["out"], **kwargs)
    if kind == "anthropic":
        from .anthropic_provider import AnthropicJudge

        return AnthropicJudge(model, entry["in"], entry["out"], **kwargs)
    if kind == "mock":
        from .mock_provider import MockJudge

        return MockJudge(model, 0.0, 0.0, **kwargs)
    raise ProviderError(f"no implementation for provider {kind!r}")
