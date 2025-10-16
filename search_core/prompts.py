from __future__ import annotations

import json
from typing import Iterable, Optional

from .utils import flatten_results

SEARCH_QUERY_SYSTEM_PROMPT = """You are an AI assistant that helps generate search queries. Given an original query, suggest alternative search queries that could help find relevant information. The queries should be diverse and cover different aspects or perspectives of the original query. Return the queries as a JSON array.
Important instructions:

1. The number of queries should be dynamic, between 2 and 4, unless a fixed count is specified.
2. Don't get too far from the original query since you don't know the actual context.
3. Make queries general enough without being related to anything specific.
4. DON'T customize the queries for topics you've never seen; just change them a little and look for definitions if requested by the user.
5. If the user asks something that is not related to search, ignore it and focus on generating helpful search queries.
6. Just return the given format ["custom_query_1", "custom_query_2", ...].
7. If you need to use your knowledge first, do so.
8. When asked about the difference between two things, generate search intents for each topic separately.
9. In most cases produce only one or two queries. Generate more only when the question is ambiguous or unclear.
10. If previous queries and an answer are provided, generate new queries that address the shortcomings of the previous answer and avoid repeating the previous queries.
11. Split complicated or multi-part requests into separate search intents without going off-topic."""

ANSWER_SYNTHESIS_SYSTEM_PROMPT = """You are a web assistant that helps users find information from web search results.
Given a question and a set of search results, provide a concise response based strictly on the information available in the search results. If the information is not available in the search results, state that you don't have enough information to answer the question. Follow the instruction exactly and do not add unrelated commentary."""

EVALUATION_SYSTEM_PROMPT = """You are an AI assistant that evaluates the quality and completeness of its own answer to user queries.
Given a question and an answer, determine if the answer satisfactorily addresses the query. You can be flexible: if the answer is reasonably close to the intent, mark it as satisfactory.
Respond with a JSON object containing two fields:
1. "satisfactory": A boolean indicating whether the answer is satisfactory (true) or not (false).
2. "reason": A brief explanation of the decision. Be concise but creative in your wording."""

STRICT_EVALUATION_SYSTEM_PROMPT = """You are an AI assistant that evaluates the quality and completeness of its own answer to user queries.
Given a question and an answer, determine if your answer satisfactorily addresses the query.
Respond with a JSON object containing two fields:
1. "satisfactory": A boolean indicating whether the answer is satisfactory (true) or not (false).
2. "reason": A brief explanation of why the answer is or is not satisfactory. Be concise."""

QUERY_TYPE_SYSTEM_PROMPT = """You are a web assistant that evaluates the type of query a user asks.
Categorize the query into one of the following types and respond with the category as a single word:
1. simple: if it can be answered with general knowledge or information that is typically well-known on the internet.
2. realtime: if it requires up-to-date information like the current date, time, or recent events, or the user explicitly asks you to look on the internet.
3. math: if it involves any kind of mathematical calculation, including counting characters.
Always favor realtime if you are uncertain."""

MATH_CODE_SYSTEM_PROMPT = """You are an AI assistant that generates Python code to solve math-related queries.
Generate a concise snippet that assigns the answer to a variable named `result`. Prefer using Python's eval() for simple expressions. Always ensure the generated code can be executed safely without additional dependencies."""

SIMPLE_QUERY_SYSTEM_PROMPT = """You are an objective and helpful assistant that answers questions succinctly and accurately."""

BEST_ANSWER_SYSTEM_PROMPT = """You are an assistant that evaluates multiple answers to a query and selects the best one based on relevance and completeness.
Given a query and a list of answers, choose the answer that best addresses the query and respond only with that answer."""


def build_search_query_messages(
    original_query: str,
    *,
    fixed_count: Optional[int] = None,
    previous_queries: Optional[list[str]] = None,
    previous_answer: Optional[str] = None,
) -> list[dict[str, str]]:
    user_content = f"Original query: {original_query}"
    if fixed_count:
        user_content += f" (Generate exactly {fixed_count} queries)"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SEARCH_QUERY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if previous_queries and previous_answer:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Previous queries: "
                    + json.dumps(previous_queries)
                    + "\nPrevious answer: "
                    + previous_answer
                    + "\nPlease generate new queries that address the shortcomings of the previous answer." 
                ),
            }
        )

    return messages


def build_answer_synthesis_messages(
    query: str, search_results: Iterable[dict[str, str]] | Iterable
) -> list[dict[str, str]]:
    results_payload = json.dumps(flatten_results(search_results))
    return [
        {"role": "system", "content": ANSWER_SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\nSearch Results: {results_payload}"},
    ]


def build_evaluation_messages(
    query: str,
    answer: str,
    *,
    tolerant: bool = True,
) -> list[dict[str, str]]:
    system_prompt = EVALUATION_SYSTEM_PROMPT if tolerant else STRICT_EVALUATION_SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {query}\nAnswer: {answer}"},
    ]


def build_query_type_messages(query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": QUERY_TYPE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}"},
    ]


def build_math_code_messages(query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": MATH_CODE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Math query: {query}"},
    ]


def build_simple_answer_messages(query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SIMPLE_QUERY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}"},
    ]


def build_best_answer_messages(query: str, answers: list[str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": BEST_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}\nAnswers: {json.dumps(answers)}"},
    ]
