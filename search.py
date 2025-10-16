from __future__ import annotations

import argparse
import sys

from search_core.config import Settings
from search_core.groq import GroqClient
from search_core.searchers import SearchScraper
from search_core.types import EvaluationResult
from search_core.workflows import SearchWorkflow


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive web search assistant powered by Groq's API",
    )
    parser.add_argument("--api-key", help="Groq API key to use for requests")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=None,
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        help="Maximum number of refinement iterations per query",
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        help="Number of search results to fetch for each generated query",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="HTTP timeout (in seconds) for outbound requests",
    )
    parser.add_argument(
        "--model",
        help="Override the Groq model to use for completions",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Set the sampling temperature for Groq completions",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Set the maximum number of tokens Groq can generate",
    )
    return parser.parse_args()


def build_workflow(args: argparse.Namespace) -> SearchWorkflow:
    settings = Settings.from_env(
        groq_api_key=args.api_key,
        debug_mode=args.debug,
        max_retries=args.max_retries,
        search_results_per_query=args.results_per_query,
        request_timeout=args.timeout,
        groq_model=args.model,
        groq_temperature=args.temperature,
        groq_max_tokens=args.max_tokens,
    ).require_api_key()

    groq_client = GroqClient(
        settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.groq_temperature,
        max_tokens=settings.groq_max_tokens,
        timeout=settings.request_timeout,
    )
    searcher = SearchScraper(settings)
    return SearchWorkflow(
        settings=settings,
        groq_client=groq_client,
        searcher=searcher,
        tolerant_evaluation=False,
    )


def print_progress(evaluation: EvaluationResult, attempt: int) -> None:
    reason = evaluation.reason or "Trying a different approach..."
    print(f"✗ Attempt {attempt}: {reason}")


def run_interactive(workflow: SearchWorkflow) -> int:
    print("Type 'quit' or 'exit' to leave the assistant.\n")

    while True:
        try:
            query = input("❖ Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0

        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            print("Goodbye!")
            return 0

        result = workflow.answer_query(query, progress_callback=print_progress)
        answer = result.answer or "No suitable answer could be produced."

        if result.evaluation.satisfactory:
            print(f"⌾ {answer}\n")
        else:
            # Provide whatever best answer we have even if not marked satisfactory.
            if result.answers_history:
                print(f"⌾ {answer}\n")
            else:
                print("✗ Unable to find a satisfactory answer.\n")


def main() -> int:
    args = parse_arguments()
    try:
        workflow = build_workflow(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return run_interactive(workflow)


if __name__ == "__main__":
    raise SystemExit(main())
