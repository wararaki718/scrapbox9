import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import requests
from ollama import ResponseError

from agent import create_model, create_support_graph
from database import ensure_database
from evaluation import run_evaluation_suite, run_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="langchain-agent-evaluation",
        description="Local-first LangGraph music-store agent evaluation scaffold.",
    )
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--database", default="chinook.db")
    parser.add_argument("--langsmith-tracing", action="store_true")
    return parser


def run_single_question(graph: object, question: str) -> str:
    result = asyncio.run(run_graph(graph, question))
    response = result.get("response", "")
    if isinstance(response, str):
        return response.strip()
    return str(response)


def _ollama_help_message(exc: BaseException) -> str:
    return (
        f"Could not reach Ollama: {exc}\n"
        "Start Ollama with: ollama serve\n"
        "Download the model with: ollama pull qwen3:1.7b"
    )


def _is_missing_model_error(exc: ResponseError) -> bool:
    status_code = getattr(exc, "status_code", None)
    return status_code == 404 and "model" in str(exc).lower()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    if args.question is None and not args.evaluate:
        parser.print_help()
        return 0

    if args.langsmith_tracing:
        if not os.environ.get("LANGSMITH_API_KEY"):
            parser.error("--langsmith-tracing requires LANGSMITH_API_KEY to already be set.")
        os.environ["LANGSMITH_TRACING"] = "true"

    try:
        database_path = ensure_database(Path(args.database))
    except RuntimeError as exc:
        cause = exc.__cause__
        if isinstance(cause, (requests.RequestException, OSError)):
            parser.error(f"Could not prepare Chinook database: {cause}")
        raise
    except (requests.RequestException, OSError) as exc:
        parser.error(f"Could not prepare Chinook database: {exc}")

    try:
        graph = create_support_graph(database_path)
        if args.question is not None:
            print(run_single_question(graph, args.question))
        if args.evaluate:
            results = asyncio.run(run_evaluation_suite(graph, create_model()))
            for result in results:
                print(f"Evaluation: {result['name']}")
                print(f"Question: {result['question']}")
                print(f"Response: {result['response']}")
                print(f"Response correct: {'PASS' if result['response_correct'] else 'FAIL'}")
                if result["route_correct"]:
                    print(f"Route: PASS ({result['route']})")
                else:
                    print(f"Route: FAIL ({result['route']} != {result['expected_route']})")
                print(f"Trajectory score: {float(result['trajectory_score']):.2f}")
                print(f"Reasoning: {result['response_reasoning']}")
                print()
    except ConnectionError as exc:
        parser.error(_ollama_help_message(exc))
    except ResponseError as exc:
        if _is_missing_model_error(exc):
            parser.error(_ollama_help_message(exc))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
