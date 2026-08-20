import sys
from pathlib import Path

import httpx
import pytest
import requests
from ollama import ResponseError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main


def test_parser_defaults_to_safe_single_question_mode() -> None:
    args = main.build_parser().parse_args([])

    assert args.question is None
    assert args.evaluate is False
    assert args.database == "chinook.db"
    assert args.langsmith_tracing is False


def test_run_single_question_normalizes_string_response(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_graph(graph: object, question: str) -> dict[str, object]:
        assert graph == "graph"
        assert question == "albums by Prince"
        return {"response": "  Purple Rain  "}

    monkeypatch.setattr(main, "run_graph", fake_run_graph)

    assert main.run_single_question("graph", "albums by Prince") == "Purple Rain"


def test_run_single_question_stringifies_non_string_response(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_graph(graph: object, question: str) -> dict[str, object]:
        assert graph == "graph"
        assert question == "invoice 237"
        return {"response": 237}

    monkeypatch.setattr(main, "run_graph", fake_run_graph)

    assert main.run_single_question("graph", "invoice 237") == "237"


def test_main_prints_help_and_returns_zero_when_no_action_is_requested(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["main.py"])

    assert main.main() == 0

    captured = capsys.readouterr()
    assert "usage:" in captured.out
    assert "--question" in captured.out
    assert captured.err == ""


def test_main_runs_single_question_and_prints_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, object] = {}
    order: list[str] = []
    graph = object()

    def fake_ensure_database(path: Path) -> Path:
        order.append("ensure_database")
        observed["database"] = path
        return path

    def fake_create_support_graph(database: Path) -> object:
        order.append("create_support_graph")
        observed["graph_database"] = database
        return graph

    def fake_run_single_question(received_graph: object, question: str) -> str:
        order.append("run_single_question")
        observed["graph"] = received_graph
        observed["question"] = question
        return "answer"

    monkeypatch.setattr(main, "ensure_database", fake_ensure_database)
    monkeypatch.setattr(main, "create_support_graph", fake_create_support_graph)
    monkeypatch.setattr(main, "run_single_question", fake_run_single_question)
    monkeypatch.setattr("sys.argv", ["main.py", "--question", "albums by Prince", "--database", "custom.db"])

    assert main.main() == 0

    captured = capsys.readouterr()
    assert observed == {
        "database": Path("custom.db"),
        "graph_database": Path("custom.db"),
        "graph": graph,
        "question": "albums by Prince",
    }
    assert order == ["ensure_database", "create_support_graph", "run_single_question"]
    assert captured.out == "answer\n"
    assert captured.err == ""


def test_main_runs_evaluation_and_formats_each_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    order: list[str] = []
    graph = object()
    judge = object()

    def fake_ensure_database(path: Path) -> Path:
        order.append("ensure_database")
        assert path == Path("eval.db")
        return path

    def fake_create_support_graph(database: Path) -> object:
        order.append("create_support_graph")
        assert database == Path("eval.db")
        return graph

    def fake_create_model() -> object:
        order.append("create_model")
        return judge

    async def fake_run_evaluation_suite(received_graph: object, received_judge: object) -> list[dict[str, object]]:
        order.append("run_evaluation_suite")
        assert received_graph is graph
        assert received_judge is judge
        return [
            {
                "name": "james-brown-lookup",
                "question": "What James Brown songs do you have?",
                "response": "Found James Brown tracks.",
                "response_correct": True,
                "response_reasoning": "Matched the catalog facts.",
                "trajectory_score": 1.0,
                "route": "question_answering_agent",
                "expected_route": "question_answering_agent",
                "route_correct": True,
            }
        ]

    monkeypatch.setattr(main, "ensure_database", fake_ensure_database)
    monkeypatch.setattr(main, "create_support_graph", fake_create_support_graph)
    monkeypatch.setattr(main, "create_model", fake_create_model)
    monkeypatch.setattr(main, "run_evaluation_suite", fake_run_evaluation_suite)
    monkeypatch.setattr("sys.argv", ["main.py", "--evaluate", "--database", "eval.db"])

    assert main.main() == 0

    captured = capsys.readouterr()
    assert order == ["ensure_database", "create_support_graph", "create_model", "run_evaluation_suite"]
    assert "Evaluation: james-brown-lookup" in captured.out
    assert "Question: What James Brown songs do you have?" in captured.out
    assert "Response correct: PASS" in captured.out
    assert "Route: PASS (question_answering_agent)" in captured.out
    assert "Trajectory score: 1.00" in captured.out
    assert captured.err == ""


def test_main_runs_question_before_evaluation_when_both_flags_are_set(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    order: list[str] = []
    graph = object()
    judge = object()

    def fake_ensure_database(path: Path) -> Path:
        order.append("ensure_database")
        return path

    def fake_create_support_graph(database: Path) -> object:
        order.append("create_support_graph")
        return graph

    def fake_run_single_question(received_graph: object, question: str) -> str:
        order.append("run_single_question")
        assert received_graph is graph
        assert question == "refund invoice 237"
        return "question answer"

    def fake_create_model() -> object:
        order.append("create_model")
        return judge

    async def fake_run_evaluation_suite(received_graph: object, received_judge: object) -> list[dict[str, object]]:
        order.append("run_evaluation_suite")
        assert received_graph is graph
        assert received_judge is judge
        return []

    monkeypatch.setattr(main, "ensure_database", fake_ensure_database)
    monkeypatch.setattr(main, "create_support_graph", fake_create_support_graph)
    monkeypatch.setattr(main, "run_single_question", fake_run_single_question)
    monkeypatch.setattr(main, "create_model", fake_create_model)
    monkeypatch.setattr(main, "run_evaluation_suite", fake_run_evaluation_suite)
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--question", "refund invoice 237", "--evaluate"],
    )

    assert main.main() == 0

    captured = capsys.readouterr()
    assert order == [
        "ensure_database",
        "create_support_graph",
        "run_single_question",
        "create_model",
        "run_evaluation_suite",
    ]
    assert captured.out.startswith("question answer\n")


def test_main_rejects_tracing_without_langsmith_api_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.setattr(main, "ensure_database", lambda path: (_ for _ in ()).throw(AssertionError("should not run")))
    monkeypatch.setattr("sys.argv", ["main.py", "--question", "hi", "--langsmith-tracing"])

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "--langsmith-tracing requires LANGSMITH_API_KEY" in captured.err
    assert "LANGSMITH_TRACING" not in main.os.environ


def test_main_enables_tracing_when_langsmith_api_key_is_present(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.setattr(main, "ensure_database", lambda path: path)
    monkeypatch.setattr(main, "create_support_graph", lambda database: object())
    monkeypatch.setattr(main, "run_single_question", lambda graph, question: "traced answer")
    monkeypatch.setattr("sys.argv", ["main.py", "--question", "hi", "--langsmith-tracing"])

    assert main.main() == 0

    captured = capsys.readouterr()
    assert captured.out == "traced answer\n"
    assert main.os.environ["LANGSMITH_TRACING"] == "true"


def test_main_reports_ollama_help_with_parser_error_for_connection_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(main, "ensure_database", lambda path: path)
    monkeypatch.setattr(main, "create_support_graph", lambda database: object())
    def fake_run_single_question(graph: object, question: str) -> str:
        request = httpx.Request("POST", "http://localhost:11434/api/generate")
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(main, "run_single_question", fake_run_single_question)
    monkeypatch.setattr("sys.argv", ["main.py", "--question", "hi"])

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "Could not reach Ollama: connection refused" in captured.err
    assert "ollama serve" in captured.err
    assert "ollama pull qwen3:1.7b" in captured.err


def test_main_does_not_obscure_non_connection_ollama_response_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "ensure_database", lambda path: path)
    monkeypatch.setattr(main, "create_support_graph", lambda database: object())
    monkeypatch.setattr(
        main,
        "run_single_question",
        lambda graph, question: (_ for _ in ()).throw(ResponseError("bad request", 400)),
    )
    monkeypatch.setattr("sys.argv", ["main.py", "--question", "hi"])

    with pytest.raises(ResponseError, match="bad request"):
        main.main()


def test_main_reports_database_download_failures_with_parser_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        main,
        "ensure_database",
        lambda path: (_ for _ in ()).throw(requests.RequestException("network dropped")),
    )
    monkeypatch.setattr("sys.argv", ["main.py", "--question", "hi"])

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "Could not prepare Chinook database: network dropped" in captured.err
