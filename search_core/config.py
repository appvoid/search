from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

from dotenv import load_dotenv


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _to_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _to_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Central configuration for the search application."""

    groq_api_key: str
    debug_mode: bool = False
    max_retries: int = 3
    search_results_per_query: int = 2
    request_timeout: float = 10.0
    max_content_length: int = 2048
    query_generation_attempts: int = 3
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.5
    groq_max_tokens: int = 1024

    @classmethod
    def from_env(cls, **overrides: Any) -> "Settings":
        """Create a :class:`Settings` instance from environment variables.

        Environment variables:
            - GROQ_API_KEY      (required for Groq API access)
            - DEBUG_MODE        (true/false)
            - MAX_RETRIES       (int)
            - SEARCH_RESULTS_PER_QUERY (int)
            - REQUEST_TIMEOUT   (float seconds)
            - MAX_CONTENT_LENGTH (int)
            - QUERY_GENERATION_ATTEMPTS (int)
            - GROQ_MODEL        (str)
            - GROQ_TEMPERATURE  (float)
            - GROQ_MAX_TOKENS   (int)
        """

        load_dotenv()

        data = {
            "groq_api_key": overrides.get("groq_api_key")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("GROQ_API")
            or "",
            "debug_mode": overrides.get("debug_mode")
            if overrides.get("debug_mode") is not None
            else _to_bool(os.getenv("DEBUG_MODE"), False),
            "max_retries": _to_int(
                overrides.get("max_retries"), _to_int(os.getenv("MAX_RETRIES"), 3)
            ),
            "search_results_per_query": _to_int(
                overrides.get("search_results_per_query"),
                _to_int(os.getenv("SEARCH_RESULTS_PER_QUERY"), 2),
            ),
            "request_timeout": _to_float(
                overrides.get("request_timeout"),
                _to_float(os.getenv("REQUEST_TIMEOUT"), 10.0),
            ),
            "max_content_length": _to_int(
                overrides.get("max_content_length"),
                _to_int(os.getenv("MAX_CONTENT_LENGTH"), 2048),
            ),
            "query_generation_attempts": _to_int(
                overrides.get("query_generation_attempts"),
                _to_int(os.getenv("QUERY_GENERATION_ATTEMPTS"), 3),
            ),
            "groq_model": overrides.get("groq_model")
            or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "groq_temperature": _to_float(
                overrides.get("groq_temperature"),
                _to_float(os.getenv("GROQ_TEMPERATURE"), 0.5),
            ),
            "groq_max_tokens": _to_int(
                overrides.get("groq_max_tokens"),
                _to_int(os.getenv("GROQ_MAX_TOKENS"), 1024),
            ),
        }

        return cls(**data)

    def require_api_key(self) -> "Settings":
        """Ensure the Groq API key is present, raising a helpful error otherwise."""

        if not self.groq_api_key:
            raise RuntimeError(
                "Groq API key is missing. Provide it via the GROQ_API_KEY environment "
                "variable or the --api-key command line option."
            )
        return self

    def with_overrides(self, **overrides: Any) -> "Settings":
        """Return a copy of the settings object overriding selected fields."""

        filtered = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **filtered)
