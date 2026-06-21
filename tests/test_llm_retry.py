"""P1: LLM rate-limit backoff. A concurrent fleet got 429s and ~65% of ingests
failed; _generate_retry turns transient rate-limit/timeout errors into
slightly-slower successes."""
import os
import pytest

from services.llm.base import LLMProvider
from services.llm.exceptions import RateLimitException, APIException


class _FlakyProvider(LLMProvider):
    """Raises `fail_times` rate-limit errors, then returns 'ok'."""
    def __init__(self, fail_times: int, exc=RateLimitException):
        self.calls = 0
        self._fail = fail_times
        self._exc = exc

    async def generate(self, prompt, temperature=None, max_tokens=None):
        self.calls += 1
        if self.calls <= self._fail:
            raise self._exc("boom")
        return "ok"

    async def validate_config(self): return True
    def get_model_info(self): return {}


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch):
    monkeypatch.setenv("LLM_RETRY_BASE_SECONDS", "0")  # no real sleeping
    monkeypatch.setenv("LLM_MAX_RETRIES", "4")


async def test_retries_then_succeeds():
    p = _FlakyProvider(fail_times=2)
    assert await p._generate_retry("x") == "ok"
    assert p.calls == 3  # 2 failures + 1 success


async def test_timeout_errors_also_retried():
    p = _FlakyProvider(fail_times=1, exc=APIException)
    assert await p._generate_retry("x") == "ok"


async def test_reraises_after_exhausting(monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    p = _FlakyProvider(fail_times=99)
    with pytest.raises(RateLimitException):
        await p._generate_retry("x")
    assert p.calls == 3  # initial + 2 retries
