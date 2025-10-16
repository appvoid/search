from __future__ import annotations

from typing import Optional

import aiohttp
from flask import Flask, jsonify, request

from search_core.config import Settings
from search_core.groq import AsyncGroqClient, GroqAPIError
from search_core.searchers import AsyncSearchScraper
from search_core.workflows import AsyncSearchWorkflow

app = Flask(__name__)

_SETTINGS: Optional[Settings] = None
_WORKFLOW: Optional[AsyncSearchWorkflow] = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings.from_env()
    return _SETTINGS


def get_workflow() -> AsyncSearchWorkflow:
    global _WORKFLOW
    if _WORKFLOW is None:
        settings = get_settings()
        if not settings.groq_api_key:
            raise RuntimeError(
                "Groq API key is missing. Set the GROQ_API_KEY environment variable."
            )
        _WORKFLOW = AsyncSearchWorkflow(
            settings=settings,
            groq_client=AsyncGroqClient(
                settings.groq_api_key,
                model=settings.groq_model,
                temperature=settings.groq_temperature,
                max_tokens=settings.groq_max_tokens,
                timeout=settings.request_timeout,
            ),
            searcher=AsyncSearchScraper(settings),
        )
    return _WORKFLOW


def parse_max_retries(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        retries = int(value)
    except (TypeError, ValueError):
        raise ValueError("max_retries must be an integer") from None
    if retries <= 0:
        raise ValueError("max_retries must be greater than zero")
    return retries


async def execute_query(query: str, max_retries: Optional[int]) -> dict[str, str]:
    workflow = get_workflow()
    settings = get_settings()
    timeout = aiohttp.ClientTimeout(total=settings.request_timeout)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        return await workflow.process_query(session, query, max_retries=max_retries)


@app.route("/ask", methods=["POST"])
async def process_query():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Query not provided"}), 400

    try:
        max_retries = parse_max_retries(data.get("max_retries"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        result = await execute_query(query, max_retries)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except GroqAPIError as exc:
        return jsonify({"error": str(exc)}), 502
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.config["DEBUG"] = get_settings().debug_mode
    app.run(debug=app.config["DEBUG"])
