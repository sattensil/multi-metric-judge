from __future__ import annotations

import sys
from types import SimpleNamespace

from packing.providers.gemini_provider import GeminiJudge


def test_gemini_accepts_live_retry_and_timeout_options(monkeypatch):
    captured: dict[str, object] = {}

    class HttpRetryOptions:
        def __init__(self, *, attempts: int):
            self.attempts = attempts

    class HttpOptions:
        def __init__(self, *, timeout: int, retry_options: HttpRetryOptions):
            self.timeout = timeout
            self.retry_options = retry_options

    def client(**kwargs):
        captured.update(kwargs)
        return object()

    genai = SimpleNamespace(
        Client=client,
        types=SimpleNamespace(
            HttpOptions=HttpOptions,
            HttpRetryOptions=HttpRetryOptions,
        ),
    )
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=genai))

    GeminiJudge(
        "gemini-2.5-flash",
        0.10,
        0.40,
        timeout=37.5,
        max_retries=4,
    )

    options = captured["http_options"]
    assert options.timeout == 37_500
    assert options.retry_options.attempts == 5
