from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional


def truncate_text(text: str | None, max_length: int) -> str | None:
    if text is None:
        return None
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def get_title_from_url(url: str) -> str:
    sanitized = re.sub(r"https?:\\/\\/(www\\.)?", "", url)
    parts = [part for part in sanitized.split("/") if part]
    if not parts:
        return sanitized
    title = parts[-1]
    title = title.replace("-", " ").replace("_", " ")
    return title.title()


def extract_choice_text(response: dict[str, Any]) -> Optional[str]:
    """Safely extract the assistant message text from a Groq API response."""

    try:
        choices = response.get("choices", [])
        if not choices:
            return None
        message = choices[0]["message"]
        return message.get("content")
    except (KeyError, TypeError, AttributeError):
        return None


def safe_json_loads(value: str) -> Any:
    """Attempt to load JSON returning ``None`` when parsing fails."""

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def flatten_results(results: Iterable["SearchResult"]) -> list[dict[str, Any]]:
    from .types import SearchResult  # Local import to avoid circular dependencies

    flattened: list[dict[str, Any]] = []
    for result in results:
        if isinstance(result, SearchResult):
            flattened.append(result.to_dict())
        elif isinstance(result, dict):
            flattened.append(result)
    return flattened
