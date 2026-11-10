"""Shared plumbing for the ScrapeUnblocker tools: the API key, the SDK clients
and turning SDK errors into text an agent can act on."""

from __future__ import annotations

import os
from typing import Any, Optional

from langchain_core.tools import BaseTool, ToolException
from pydantic import Field, PrivateAttr, SecretStr, model_validator
from scrapeunblocker import AsyncClient, Client
from scrapeunblocker.exceptions import (
    AuthenticationError,
    BlockedError,
    PaymentRequiredError,
    RateLimitError,
    ScrapeTimeoutError,
    ScrapeUnblockerError,
    TargetNotFoundError,
)

from langchain_scrapeunblocker.version import __version__

DEFAULT_BASE_URL = "https://api.scrapeunblocker.com"
# Checked in this order; the SDK itself reads only the second.
API_KEY_ENV_VARS = ("SCRAPEUNBLOCKER_API_KEY", "SCRAPEUNBLOCKER_KEY")
SIGNUP_URL = (
    "https://app.scrapeunblocker.com"
    "?utm_source=langchain&utm_medium=integration&utm_campaign=langchain"
)


def _key_from_env() -> Optional[str]:
    for name in API_KEY_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def cap_text(text: str, max_chars: int, hint: str) -> str:
    """Cut `text` to `max_chars`, saying so, so the agent knows it is partial."""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n\n[Cut at {max_chars} of {len(text)} characters. {hint}]"


def error_text(err: BaseException) -> str:
    """What the agent is told when a call fails, with what to do next."""
    if isinstance(err, TargetNotFoundError):
        return (
            f"The page itself answered {err.origin_status} (not found or gone). "
            "It was billed as one request; do not retry the same URL."
        )
    if isinstance(err, AuthenticationError):
        return (
            "ScrapeUnblocker rejected the API key, or the account has no active plan. "
            f"Check the key or pick a plan at {SIGNUP_URL}"
        )
    if isinstance(err, PaymentRequiredError):
        return f"The ScrapeUnblocker account is out of credits or over its limit: {err}"
    if isinstance(err, BlockedError):
        return f"The site blocked this request even through ScrapeUnblocker: {err}. Not billed."
    if isinstance(err, RateLimitError):
        return "Too many requests at once for this API key. Wait a moment and try again."
    if isinstance(err, ScrapeTimeoutError):
        return "The page took too long to load. Try again, or try a simpler URL."
    return f"ScrapeUnblocker request failed: {err}"


class ScrapeUnblockerBaseTool(BaseTool):
    """Base class: owns the API key and the ScrapeUnblocker SDK clients."""

    api_key: Optional[SecretStr] = Field(default=None)
    """ScrapeUnblocker API key. Read from ``SCRAPEUNBLOCKER_API_KEY`` (or
    ``SCRAPEUNBLOCKER_KEY``) when not passed."""

    base_url: str = DEFAULT_BASE_URL
    """API host. Rarely needed."""

    timeout: float = 180.0
    """Seconds to wait for one API call. Protected pages can take a while."""

    max_retries: int = 2
    """Retries for transient failures (429, 502-504, network errors)."""

    handle_tool_error: bool = True
    """Failed calls come back to the agent as text instead of raising."""

    _client: Optional[Client] = PrivateAttr(default=None)
    _aclient: Optional[AsyncClient] = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _api_key_from_env(cls, values: Any) -> Any:
        if isinstance(values, dict) and not values.get("api_key"):
            key = _key_from_env()
            if key:
                values = {**values, "api_key": key}
        return values

    def _key(self) -> str:
        if self.api_key is None or not self.api_key.get_secret_value():
            raise ToolException(
                "No ScrapeUnblocker API key. Pass api_key=... or set SCRAPEUNBLOCKER_API_KEY. "
                f"Get a key at {SIGNUP_URL}"
            )
        return self.api_key.get_secret_value()

    @staticmethod
    def _name_this_package(client: Any) -> None:
        # The API counts calls per client library by User-Agent token.
        try:
            headers = client._http.headers
            headers["User-Agent"] = f"{headers.get('User-Agent', '')} langchain-scrapeunblocker/{__version__}".strip()
        except AttributeError:
            pass

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(
                self._key(), base_url=self.base_url, timeout=self.timeout, max_retries=self.max_retries
            )
            self._name_this_package(self._client)
        return self._client

    @property
    def aclient(self) -> AsyncClient:
        if self._aclient is None:
            self._aclient = AsyncClient(
                self._key(), base_url=self.base_url, timeout=self.timeout, max_retries=self.max_retries
            )
            self._name_this_package(self._aclient)
        return self._aclient

    @staticmethod
    def _fail(err: ScrapeUnblockerError) -> ToolException:
        return ToolException(error_text(err))
