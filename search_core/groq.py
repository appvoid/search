from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable, Optional

import aiohttp
import requests


def _format_error(status: int, payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            detail = error.get("message") or error
        elif error:
            detail = error
        else:
            detail = payload
        return f"Groq API HTTP {status}: {detail}"
    return f"Groq API HTTP {status}: {payload}"


class GroqAPIError(RuntimeError):
    """Raised when the Groq API returns an error or an unexpected payload."""


class GroqClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.5,
        max_tokens: int = 1024,
        timeout: float = 15.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.session = requests.Session()
        self.timeout = timeout
        self.default_payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "stream": False,
            "stop": None,
        }
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def chat_completion(
        self, messages: Iterable[Dict[str, str]], **overrides: Any
    ) -> dict[str, Any]:
        payload = {**self.default_payload, **overrides, "messages": list(messages)}
        try:
            response = self.session.post(
                self.base_url, headers=self.headers, json=payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise GroqAPIError(f"Groq API request failed: {exc}") from exc

        if response.status_code >= 400:
            try:
                error_payload = response.json()
            except ValueError:
                error_payload = response.text
            raise GroqAPIError(_format_error(response.status_code, error_payload))

        try:
            data = response.json()
        except ValueError as exc:
            raise GroqAPIError("Groq API response was not valid JSON") from exc

        if "choices" not in data:
            raise GroqAPIError(f"Unexpected Groq API response: {data}")

        return data

    def close(self) -> None:
        self.session.close()


class AsyncGroqClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.5,
        max_tokens: int = 1024,
        timeout: float = 15.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.timeout = timeout
        self.default_payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "stream": False,
            "stop": None,
        }
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def chat_completion(
        self,
        session: aiohttp.ClientSession,
        messages: Iterable[Dict[str, str]],
        **overrides: Any,
    ) -> dict[str, Any]:
        payload = {**self.default_payload, **overrides, "messages": list(messages)}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with session.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=timeout,
            ) as response:
                try:
                    data = await response.json()
                except aiohttp.ContentTypeError:
                    text = await response.text()
                    if response.status >= 400:
                        raise GroqAPIError(_format_error(response.status, text))
                    raise GroqAPIError("Groq API response was not valid JSON")

                if response.status >= 400:
                    raise GroqAPIError(_format_error(response.status, data))
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise GroqAPIError(f"Groq API request failed: {exc}") from exc

        if "choices" not in data:
            raise GroqAPIError(f"Unexpected Groq API response: {data}")

        return data
