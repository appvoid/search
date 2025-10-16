from __future__ import annotations

import asyncio
from typing import Callable, Iterable, Optional, Sequence

from .config import Settings
from .groq import AsyncGroqClient, GroqAPIError, GroqClient
from .prompts import (
    build_answer_synthesis_messages,
    build_best_answer_messages,
    build_evaluation_messages,
    build_math_code_messages,
    build_query_type_messages,
    build_search_query_messages,
    build_simple_answer_messages,
)
from .searchers import AsyncSearchScraper, SearchScraper
from .types import EvaluationResult, SearchResult, WorkflowResult
from .utils import extract_choice_text, safe_json_loads


class SearchWorkflow:
    """Synchronous workflow orchestrating the search experience."""

    def __init__(
        self,
        settings: Settings,
        groq_client: GroqClient,
        searcher: SearchScraper,
        *,
        tolerant_evaluation: bool = False,
    ) -> None:
        self.settings = settings
        self.client = groq_client
        self.searcher = searcher
        self.tolerant_evaluation = tolerant_evaluation

    def _log(self, message: str) -> None:
        if self.settings.debug_mode:
            print(message)

    def generate_search_queries(
        self,
        query: str,
        *,
        fixed_count: Optional[int] = None,
        previous_queries: Optional[list[str]] = None,
        previous_answer: Optional[str] = None,
    ) -> list[str]:
        messages = build_search_query_messages(
            query,
            fixed_count=fixed_count,
            previous_queries=previous_queries,
            previous_answer=previous_answer,
        )

        attempts = max(1, self.settings.query_generation_attempts)
        for attempt in range(attempts):
            try:
                response = self.client.chat_completion(messages)
            except GroqAPIError as exc:
                self._log(f"> Failed to generate queries (attempt {attempt + 1}): {exc}")
                continue

            content = extract_choice_text(response)
            if not content:
                continue

            parsed = safe_json_loads(content)
            if isinstance(parsed, list):
                queries = [str(item).strip() for item in parsed if str(item).strip()]
                if not queries:
                    continue
                if fixed_count and len(queries) != fixed_count:
                    continue
                if not fixed_count and len(queries) > 5:
                    queries = queries[:5]
                return queries

        self._log("> Falling back to the original query due to generation failures")
        return [query]

    def fetch_search_results(self, queries: Sequence[str]) -> list[SearchResult]:
        if not queries:
            return []
        return self.searcher.search_many(queries)

    def synthesize_answer(
        self, query: str, results: Iterable[SearchResult]
    ) -> Optional[str]:
        messages = build_answer_synthesis_messages(query, [r for r in results])
        try:
            response = self.client.chat_completion(messages)
        except GroqAPIError as exc:
            self._log(f"> Failed to synthesize answer: {exc}")
            return None

        content = extract_choice_text(response)
        if content:
            return content.strip()
        return None

    def evaluate_answer(self, query: str, answer: str) -> EvaluationResult:
        messages = build_evaluation_messages(
            query, answer, tolerant=self.tolerant_evaluation
        )
        try:
            response = self.client.chat_completion(messages)
        except GroqAPIError as exc:
            self._log(f"> Failed to evaluate answer: {exc}")
            return EvaluationResult(False, f"Unable to evaluate the answer: {exc}")

        content = extract_choice_text(response)
        data = safe_json_loads(content) if content else None

        if isinstance(data, dict):
            satisfactory = bool(data.get("satisfactory"))
            reason = str(data.get("reason") or "")
            return EvaluationResult(satisfactory, reason)

        return EvaluationResult(False, "Unable to evaluate the answer")

    def answer_query(
        self,
        query: str,
        *,
        progress_callback: Optional[Callable[[EvaluationResult, int], None]] = None,
    ) -> WorkflowResult:
        previous_queries: Optional[list[str]] = None
        previous_answer: Optional[str] = None
        answers_history: list[str] = []
        evaluation = EvaluationResult(False, "")

        for attempt in range(1, self.settings.max_retries + 1):
            search_queries = self.generate_search_queries(
                query,
                previous_queries=previous_queries,
                previous_answer=previous_answer,
            )
            search_results = self.fetch_search_results(search_queries)
            if not search_results:
                self._log(
                    f"> No search results found for queries: {', '.join(search_queries)}"
                )

            answer = self.synthesize_answer(query, search_results)
            if not answer:
                evaluation = EvaluationResult(
                    False, "Unable to produce an answer from the gathered search results."
                )
                break

            answers_history.append(answer)
            evaluation = self.evaluate_answer(query, answer)
            if evaluation.satisfactory:
                return WorkflowResult(
                    answer=answer,
                    evaluation=evaluation,
                    attempts=attempt,
                    answers_history=answers_history,
                )

            previous_queries = search_queries
            previous_answer = answer
            if progress_callback:
                progress_callback(evaluation, attempt)
            else:
                self._log(f"✗ {evaluation.reason}")

        final_answer = answers_history[-1] if answers_history else ""
        return WorkflowResult(
            answer=final_answer,
            evaluation=evaluation,
            attempts=self.settings.max_retries,
            answers_history=answers_history,
        )


class AsyncSearchWorkflow:
    """Asynchronous workflow orchestrating query handling for the web server."""

    def __init__(
        self,
        settings: Settings,
        groq_client: AsyncGroqClient,
        searcher: AsyncSearchScraper,
    ) -> None:
        self.settings = settings
        self.client = groq_client
        self.searcher = searcher

    def _log(self, message: str) -> None:
        if self.settings.debug_mode:
            print(message)

    async def generate_search_queries(
        self,
        session,
        query: str,
        *,
        fixed_count: Optional[int] = None,
        previous_queries: Optional[list[str]] = None,
        previous_answer: Optional[str] = None,
    ) -> list[str]:
        messages = build_search_query_messages(
            query,
            fixed_count=fixed_count,
            previous_queries=previous_queries,
            previous_answer=previous_answer,
        )

        attempts = max(1, self.settings.query_generation_attempts)
        for attempt in range(attempts):
            try:
                response = await self.client.chat_completion(session, messages)
            except GroqAPIError as exc:
                self._log(f"> Failed to generate queries (attempt {attempt + 1}): {exc}")
                continue

            content = extract_choice_text(response)
            if not content:
                continue

            parsed = safe_json_loads(content)
            if isinstance(parsed, list):
                queries = [str(item).strip() for item in parsed if str(item).strip()]
                if not queries:
                    continue
                if fixed_count and len(queries) != fixed_count:
                    continue
                if not fixed_count and len(queries) > 5:
                    queries = queries[:5]
                return queries

        self._log("> Falling back to the original query due to generation failures")
        return [query]

    async def synthesize_answer(
        self,
        session,
        query: str,
        results: Iterable[SearchResult],
    ) -> Optional[str]:
        messages = build_answer_synthesis_messages(query, [r for r in results])
        try:
            response = await self.client.chat_completion(session, messages)
        except GroqAPIError as exc:
            self._log(f"> Failed to synthesize answer: {exc}")
            return None

        content = extract_choice_text(response)
        if content:
            return content.strip()
        return None

    async def evaluate_answer(
        self, session, query: str, answer: str
    ) -> EvaluationResult:
        messages = build_evaluation_messages(query, answer, tolerant=True)
        try:
            response = await self.client.chat_completion(session, messages)
        except GroqAPIError as exc:
            self._log(f"> Failed to evaluate answer: {exc}")
            return EvaluationResult(False, f"Unable to evaluate the answer: {exc}")

        content = extract_choice_text(response)
        data = safe_json_loads(content) if content else None

        if isinstance(data, dict):
            satisfactory = bool(data.get("satisfactory"))
            reason = str(data.get("reason") or "")
            return EvaluationResult(satisfactory, reason)

        return EvaluationResult(False, "Unable to evaluate the answer")

    async def evaluate_best_answer(
        self, session, query: str, answers: list[str]
    ) -> Optional[str]:
        messages = build_best_answer_messages(query, answers)
        try:
            response = await self.client.chat_completion(session, messages)
        except GroqAPIError as exc:
            self._log(f"> Failed to evaluate best answer: {exc}")
            return None

        content = extract_choice_text(response)
        return content.strip() if content else None

    async def classify_query(self, session, query: str) -> str:
        messages = build_query_type_messages(query)
        try:
            response = await self.client.chat_completion(session, messages)
        except GroqAPIError:
            return "simple"

        content = extract_choice_text(response)
        if not content:
            return "simple"
        return content.strip().lower()

    async def handle_simple_query(self, session, query: str) -> str:
        messages = build_simple_answer_messages(query)
        response = await self.client.chat_completion(session, messages)
        content = extract_choice_text(response)
        return content.strip() if content else ""

    async def handle_math_query(self, session, query: str) -> str:
        messages = build_math_code_messages(query)
        try:
            response = await self.client.chat_completion(session, messages)
        except GroqAPIError as exc:
            return f"Unable to generate math solution: {exc}"

        code = extract_choice_text(response) or ""
        if self.settings.debug_mode:
            print("Code >", code)

        if not code:
            return "Unable to retrieve the math solution."

        try:
            return self._execute_generated_code(code)
        except Exception as exc:  # noqa: BLE001
            return f"Error while executing generated code: {exc}"

    def _execute_generated_code(self, code: str) -> str:
        allowed_builtins = {
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
            "sum": sum,
            "len": len,
            "eval": eval,
        }
        local_vars: dict[str, object] = {}
        exec(code, {"__builtins__": allowed_builtins}, local_vars)  # noqa: S102
        result = local_vars.get("result")
        if isinstance(result, str):
            return result
        return str(result)

    async def handle_search_flow(
        self,
        session,
        query: str,
        *,
        max_retries: Optional[int] = None,
    ) -> WorkflowResult:
        limit = max_retries or self.settings.max_retries
        previous_queries: Optional[list[str]] = None
        previous_answer: Optional[str] = None
        answers_history: list[str] = []
        evaluation = EvaluationResult(False, "")

        for attempt in range(1, limit + 1):
            search_queries = await self.generate_search_queries(
                session,
                query,
                previous_queries=previous_queries,
                previous_answer=previous_answer,
            )
            search_results = await self.searcher.search_many(session, search_queries)
            answer = await self.synthesize_answer(session, query, search_results)
            if not answer:
                evaluation = EvaluationResult(
                    False,
                    "Unable to produce an answer from the gathered search results.",
                )
                break

            answers_history.append(answer)
            evaluation = await self.evaluate_answer(session, query, answer)
            if evaluation.satisfactory:
                return WorkflowResult(
                    answer=answer,
                    evaluation=evaluation,
                    attempts=attempt,
                    answers_history=answers_history,
                )

            previous_queries = search_queries
            previous_answer = answer
            self._log(f"✗ {evaluation.reason}")

        final_answer = answers_history[-1] if answers_history else ""
        return WorkflowResult(
            answer=final_answer,
            evaluation=evaluation,
            attempts=limit,
            answers_history=answers_history,
        )

    async def process_query(
        self,
        session,
        query: str,
        *,
        max_retries: Optional[int] = None,
    ) -> dict[str, str]:
        query_type = await self.classify_query(session, query)

        if query_type == "math":
            result = await self.handle_math_query(session, query)
            return {"response": str(result), "type": "math"}

        if query_type == "simple":
            answer = await self.handle_simple_query(session, query)
            return {"response": answer, "type": "text"}

        workflow_result = await self.handle_search_flow(
            session, query, max_retries=max_retries
        )

        if not workflow_result.evaluation.satisfactory and workflow_result.answers_history:
            best_answer = await self.evaluate_best_answer(
                session, query, workflow_result.answers_history
            )
            if best_answer:
                workflow_result.answer = best_answer

        return {"response": workflow_result.answer, "type": "search"}
