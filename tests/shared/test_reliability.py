import httpx
from openai import AuthenticationError, RateLimitError

from src.shared.reliability import is_transient


def test_quota_is_not_transient():
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.org"))
    assert not is_transient(RateLimitError("quota", response=response, body={"code": "insufficient_quota"}))
    assert not is_transient(
        RateLimitError("credit", response=response, body={"code": "credit_balance_exhausted"})
    )
    assert is_transient(RateLimitError("rate", response=response, body={"code": "rate_limit_exceeded"}))
    assert not is_transient(
        AuthenticationError("auth", response=httpx.Response(401, request=response.request), body={})
    )
