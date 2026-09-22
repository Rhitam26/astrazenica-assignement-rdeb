"""One bounded retry policy. Permanent credit/authentication errors never retry."""

from openai import APIConnectionError, APIStatusError, APITimeoutError
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from src.shared.config import Settings


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        body = str(exc.body).lower()
        if any(
            code in body for code in ("insufficient_quota", "credit_balance_exhausted", "billing_hard_limit")
        ):
            return False
        return exc.status_code in (408, 409, 429) or exc.status_code >= 500
    return False


def retry_policy(settings: Settings) -> Retrying:
    return Retrying(
        retry=retry_if_exception(is_transient),
        stop=stop_after_attempt(settings.provider_max_attempts),
        wait=wait_random_exponential(min=1, max=5),
        reraise=True,
    )
