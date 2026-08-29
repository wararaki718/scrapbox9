# LangChain Agent Evaluation with Ollama Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable, local-first LangGraph music-store support agent that uses Ollama `qwen3:1.7b` and evaluates final responses, trajectories, and intent routing without requiring cloud credentials.

**Architecture:** A parent LangGraph routes each request to either a tool-using catalog lookup agent or a refund subgraph. Database and LangChain tools are isolated from graph construction, while the evaluation module invokes the same graph in mock-safe mode and captures graph events. The command-line entry point exposes both one-question and evaluation-suite workflows.

**Tech Stack:** Python 3.11+, LangChain, LangGraph, langchain-ollama, SQLite, requests, tabulate, pytest.

---

## File Structure

- `agent_sample/langchain-agent-evaluation/requirements.txt`: pinned-to-major runtime and test dependencies.
- `agent_sample/langchain-agent-evaluation/database.py`: database path management, atomic Chinook download, parameterized catalog/purchase queries, and mock-safe refund calculation.
- `agent_sample/langchain-agent-evaluation/schemas.py`: Pydantic structured-output models and LangGraph state typing.
- `agent_sample/langchain-agent-evaluation/tools.py`: catalog lookup tools backed by `database.py`.
- `agent_sample/langchain-agent-evaluation/refund.py`: refund extraction prompt and refund subgraph.
- `agent_sample/langchain-agent-evaluation/agent.py`: Ollama model construction, intent classifier, lookup ReAct agent, and parent graph.
- `agent_sample/langchain-agent-evaluation/evaluation.py`: tutorial-derived examples, local evaluators, streamed trajectory collection, and suite output.
- `agent_sample/langchain-agent-evaluation/main.py`: argparse CLI and user-facing error reporting.
- `agent_sample/langchain-agent-evaluation/README.md`: setup, safe commands, expected local behavior, and optional LangSmith tracing.
- `agent_sample/langchain-agent-evaluation/tests/test_database.py`: deterministic SQLite behavior tests.
- `agent_sample/langchain-agent-evaluation/tests/test_evaluation.py`: deterministic evaluator and trajectory normalization tests.

### Task 1: Create the runnable sample scaffold

**Files:**
- Create: `agent_sample/langchain-agent-evaluation/requirements.txt`
- Create: `agent_sample/langchain-agent-evaluation/README.md`
- Create: `agent_sample/langchain-agent-evaluation/main.py`

- [ ] **Step 1: Write the failing CLI smoke test**

Create `agent_sample/langchain-agent-evaluation/tests/test_main.py`:

```python
from main import build_parser


def test_parser_defaults_to_safe_single_question_mode() -> None:
    args = build_parser().parse_args([])

    assert args.question is None
    assert args.evaluate is False
    assert args.database == "chinook.db"
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `cd agent_sample/langchain-agent-evaluation && pytest tests/test_main.py::test_parser_defaults_to_safe_single_question_mode -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'main'`.

- [ ] **Step 3: Add the dependency manifest and parser implementation**

Write `requirements.txt`:

```text
langchain>=1.0,<2.0
langchain-ollama>=1.0,<2.0
langgraph>=1.0,<2.0
requests>=2.32,<3.0
tabulate>=0.9,<1.0
pytest>=8.0,<9.0
```

Write `main.py` with this parser boundary:

```python
import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local Ollama music-store support agent."
    )
    parser.add_argument("--question", help="Ask one customer-support question.")
    parser.add_argument(
        "--evaluate", action="store_true", help="Run the local evaluation suite."
    )
    parser.add_argument("--database", default="chinook.db")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.question is None and not args.evaluate:
        build_parser().print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `cd agent_sample/langchain-agent-evaluation && pytest tests/test_main.py::test_parser_defaults_to_safe_single_question_mode -v`

Expected: PASS.

- [ ] **Step 5: Document the initial setup**

Write a README section containing the exact setup and safe commands:

```markdown
## Setup

```bash
cd agent_sample/langchain-agent-evaluation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3:1.7b
```

## Run

```bash
python main.py --question "What James Brown songs do you have?"
python main.py --evaluate
```

Both commands use `env="test"` and never delete Chinook purchase records.
```

- [ ] **Step 6: Commit the scaffold**

```bash
git add agent_sample/langchain-agent-evaluation
git commit -m "feat: scaffold Ollama agent evaluation sample"
```

### Task 2: Implement safe Chinook database access

**Files:**
- Create: `agent_sample/langchain-agent-evaluation/database.py`
- Test: `agent_sample/langchain-agent-evaluation/tests/test_database.py`

- [ ] **Step 1: Write failing database tests**

Use a temporary SQLite database to define and verify the public API:

```python
import sqlite3

from database import lookup_purchases, refund


def test_refund_in_mock_mode_reports_total_without_deleting(tmp_path) -> None:
    database = tmp_path / "chinook.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE Invoice (InvoiceId INTEGER PRIMARY KEY, Total REAL NOT NULL);
        CREATE TABLE InvoiceLine (
            InvoiceLineId INTEGER PRIMARY KEY,
            InvoiceId INTEGER NOT NULL,
            UnitPrice REAL NOT NULL,
            Quantity INTEGER NOT NULL
        );
        INSERT INTO Invoice VALUES (237, 1.98);
        INSERT INTO InvoiceLine VALUES (1, 237, 0.99, 2);
        """
    )
    connection.close()

    assert refund(database, invoice_id=237, invoice_line_ids=None, mock=True) == 1.98

    connection = sqlite3.connect(database)
    assert connection.execute("SELECT COUNT(*) FROM Invoice").fetchone()[0] == 1
    connection.close()


def test_lookup_purchases_uses_customer_identity_and_optional_track(tmp_path) -> None:
    database = tmp_path / "chinook.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE Customer (
            CustomerId INTEGER PRIMARY KEY, FirstName TEXT, LastName TEXT, Phone TEXT
        );
        CREATE TABLE Invoice (
            InvoiceId INTEGER PRIMARY KEY, CustomerId INTEGER, InvoiceDate TEXT
        );
        CREATE TABLE InvoiceLine (
            InvoiceLineId INTEGER PRIMARY KEY, InvoiceId INTEGER, TrackId INTEGER,
            UnitPrice REAL, Quantity INTEGER
        );
        CREATE TABLE Track (TrackId INTEGER PRIMARY KEY, Name TEXT, AlbumId INTEGER);
        CREATE TABLE Album (AlbumId INTEGER PRIMARY KEY, Title TEXT, ArtistId INTEGER);
        CREATE TABLE Artist (ArtistId INTEGER PRIMARY KEY, Name TEXT);
        INSERT INTO Customer VALUES (1, 'Aaron', 'Mitchell', '+1 204');
        INSERT INTO Invoice VALUES (2, 1, '2009-08-06');
        INSERT INTO Artist VALUES (3, 'Led Zeppelin');
        INSERT INTO Album VALUES (4, 'IV', 3);
        INSERT INTO Track VALUES (5, 'Black Dog', 4);
        INSERT INTO InvoiceLine VALUES (6, 2, 5, 0.99, 1);
        """
    )
    connection.close()

    rows = lookup_purchases(
        database, "Aaron", "Mitchell", "+1 204", "Black Dog", None, None, None
    )

    assert rows == [{
        "invoice_line_id": 6, "track_name": "Black Dog",
        "artist_name": "Led Zeppelin", "purchase_date": "2009-08-06",
        "quantity_purchased": 1, "price_per_unit": 0.99,
    }]
```

- [ ] **Step 2: Run the database tests to verify they fail**

Run: `cd agent_sample/langchain-agent-evaluation && pytest tests/test_database.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'database'`.

- [ ] **Step 3: Implement the database module**

Implement these exact public interfaces:

```python
from pathlib import Path

DATABASE_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"

def refund(
    database: Path, invoice_id: int | None, invoice_line_ids: list[int] | None,
    *, mock: bool,
) -> float:
    with sqlite3.connect(database) as connection:
        total = _refund_total(connection, invoice_id, invoice_line_ids)
        if not mock:
            _delete_refunded_rows(connection, invoice_id, invoice_line_ids)
        return total
```

`ensure_database` must request `DATABASE_URL` with a timeout, call `raise_for_status()`, write to a sibling temporary path, then atomically replace the requested path. All query values must be passed through SQLite parameters. `refund` must calculate the total before conditionally deleting invoice lines followed by invoices, and commit only when `mock` is false.

Add `lookup_purchases`, `find_tracks`, `find_albums`, and `find_artists` as parameterized-query functions returning lists of dictionaries. `lookup_purchases` accepts the required customer identity followed by optional track, album, artist, and ISO-8601 date filters. The three catalog functions accept their filters shown in the Task 3 tools and select display-ready names from the Track/Album/Artist joins.

- [ ] **Step 4: Run the database tests to verify they pass**

Run: `cd agent_sample/langchain-agent-evaluation && pytest tests/test_database.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the database layer**

```bash
git add agent_sample/langchain-agent-evaluation/database.py agent_sample/langchain-agent-evaluation/tests/test_database.py
git commit -m "feat: add safe Chinook database access"
```

### Task 3: Define graph state and catalog tools

**Files:**
- Create: `agent_sample/langchain-agent-evaluation/schemas.py`
- Create: `agent_sample/langchain-agent-evaluation/tools.py`
- Modify: `agent_sample/langchain-agent-evaluation/tests/test_database.py`

- [ ] **Step 1: Write the failing catalog lookup test**

Append this test using the temporary schema from Task 2:

```python
from database import find_tracks


def test_find_tracks_returns_track_album_and_artist(tmp_path) -> None:
    database = build_catalog_database(tmp_path)

    assert find_tracks(database, name="Black Dog", artist="Led Zeppelin") == [{
        "track_name": "Black Dog", "album_title": "IV", "artist_name": "Led Zeppelin"
    }]
```

Extract the repeated database setup from Task 2 into `build_catalog_database(tmp_path)` in the same test module.

- [ ] **Step 2: Run the selected test to verify it fails**

Run: `cd agent_sample/langchain-agent-evaluation && pytest tests/test_database.py::test_find_tracks_returns_track_album_and_artist -v`

Expected: FAIL because `find_tracks` has not yet implemented the catalog join.

- [ ] **Step 3: Add typed schemas and tools**

Define `AgentState` as a `TypedDict` with `messages: Annotated[list[AnyMessage], add_messages]`, nullable `followup`, nullable invoice/customer/catalog fields, and `route: Literal["refund_agent", "question_answering_agent"] | None`. Define Pydantic `PurchaseInformation` and `UserIntent` models with the same extraction fields and route literal.

In `tools.py`, expose these LangChain tools, each returning JSON text from the corresponding database function:

```python
@tool
def lookup_track(name: str | None = None, artist: str | None = None) -> str:
    return json.dumps(find_tracks(database, name, artist))

@tool
def lookup_album(title: str | None = None, artist: str | None = None) -> str:
    return json.dumps(find_albums(database, title, artist))

@tool
def lookup_artist(name: str) -> str:
    return json.dumps(find_artists(database, name))
```

Use a `create_catalog_tools(database: Path)` factory so each tool closes over the configured database path rather than relying on process-global mutable state.

- [ ] **Step 4: Implement the catalog join and run tests**

Implement exact-match, case-insensitive filters in `find_tracks`, `find_albums`, and `find_artists`; return rows ordered by display name and limited to 20 records. Run:

`cd agent_sample/langchain-agent-evaluation && pytest tests/test_database.py -v`

Expected: PASS.

- [ ] **Step 5: Commit state and tools**

```bash
git add agent_sample/langchain-agent-evaluation/schemas.py agent_sample/langchain-agent-evaluation/tools.py agent_sample/langchain-agent-evaluation/database.py agent_sample/langchain-agent-evaluation/tests/test_database.py
git commit -m "feat: add catalog tools and agent schemas"
```

### Task 4: Build the mock-safe refund subgraph

**Files:**
- Create: `agent_sample/langchain-agent-evaluation/refund.py`
- Create: `agent_sample/langchain-agent-evaluation/tests/test_refund.py`

- [ ] **Step 1: Write failing pure refund-routing tests**

Make extraction routing independently testable:

```python
from refund import next_refund_step


def test_next_refund_step_prefers_refund_for_invoice() -> None:
    assert next_refund_step(invoice_id=237, invoice_line_ids=None, first_name=None,
                            last_name=None, phone=None) == "refund"


def test_next_refund_step_uses_lookup_for_complete_identity() -> None:
    assert next_refund_step(invoice_id=None, invoice_line_ids=None, first_name="Aaron",
                            last_name="Mitchell", phone="+1 204") == "lookup"


def test_next_refund_step_requests_details_when_identity_is_incomplete() -> None:
    assert next_refund_step(invoice_id=None, invoice_line_ids=None, first_name="Aaron",
                            last_name=None, phone=None) == "respond"
```

- [ ] **Step 2: Run the routing tests to verify they fail**

Run: `cd agent_sample/langchain-agent-evaluation && pytest tests/test_refund.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'refund'`.

- [ ] **Step 3: Implement refund graph construction**

Implement:

```python
def next_refund_step(
    invoice_id: int | None, invoice_line_ids: list[int] | None,
    first_name: str | None, last_name: str | None, phone: str | None,
) -> Literal["refund", "lookup", "respond"]:
    if invoice_id is not None or invoice_line_ids:
        return "refund"
    if all((first_name, last_name, phone)):
        return "lookup"
    return "respond"
```

Add `create_refund_graph(model: ChatOllama, database: Path) -> CompiledStateGraph`. Its `gather_info` node invokes `model.with_structured_output(PurchaseInformation)` with explicit store/refund instructions, merges only parsed values into the state, then routes with `next_refund_step`. `lookup` calls `lookup_purchases` and renders either a no-purchases message or a `tabulate` table. `refund` reads `config["configurable"]["env"]`; it passes `mock=True` unless the value is exactly `"prod"`. `respond` writes the extraction model's follow-up or a fixed request for first name, last name, and phone number.

- [ ] **Step 4: Run the refund tests to verify they pass**

Run: `cd agent_sample/langchain-agent-evaluation && pytest tests/test_refund.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the refund flow**

```bash
git add agent_sample/langchain-agent-evaluation/refund.py agent_sample/langchain-agent-evaluation/tests/test_refund.py
git commit -m "feat: add safe refund subgraph"
```

### Task 5: Compose the Ollama parent graph

**Files:**
- Create: `agent_sample/langchain-agent-evaluation/agent.py`
- Create: `agent_sample/langchain-agent-evaluation/tests/test_agent.py`

- [ ] **Step 1: Write failing intent-normalization tests**

Keep model I/O outside the test by testing the deterministic conversion:

```python
from agent import normalize_route


def test_normalize_route_maps_refund_intent() -> None:
    assert normalize_route("refund") == "refund_agent"


def test_normalize_route_maps_catalog_intent() -> None:
    assert normalize_route("question_answering") == "question_answering_agent"
```

- [ ] **Step 2: Run the agent tests to verify they fail**

Run: `cd agent_sample/langchain-agent-evaluation && pytest tests/test_agent.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'agent'`.

- [ ] **Step 3: Implement model and graph composition**

Implement:

```python
def create_model() -> ChatOllama:
    return ChatOllama(model="qwen3:1.7b", temperature=0)


def normalize_route(intent: Literal["refund", "question_answering"]) -> str:
    return f"{intent}_agent"
```

Add `create_support_graph(database: Path) -> CompiledStateGraph`. It builds a structured `UserIntent` classifier, a `create_agent` catalog subagent using `create_catalog_tools(database)`, and `create_refund_graph`. Add `intent_classifier`, `refund_agent`, `question_answering_agent`, and `compile_followup` nodes to a `StateGraph`. `compile_followup` uses the existing `followup` or converts the final message content to a string. The parent graph must have the entry point `intent_classifier` and terminal edge after `compile_followup`.

- [ ] **Step 4: Run agent tests to verify they pass**

Run: `cd agent_sample/langchain-agent-evaluation && pytest tests/test_agent.py -v`

Expected: PASS.

- [ ] **Step 5: Commit parent graph composition**

```bash
git add agent_sample/langchain-agent-evaluation/agent.py agent_sample/langchain-agent-evaluation/tests/test_agent.py
git commit -m "feat: compose Ollama support agent graph"
```

### Task 6: Add deterministic evaluation primitives and local evaluation suite

**Files:**
- Create: `agent_sample/langchain-agent-evaluation/evaluation.py`
- Create: `agent_sample/langchain-agent-evaluation/tests/test_evaluation.py`

- [ ] **Step 1: Write failing evaluator tests**

```python
from evaluation import route_is_correct, trajectory_subsequence


def test_trajectory_subsequence_scores_ordered_expected_steps() -> None:
    assert trajectory_subsequence(
        ["intent_classifier", "question_answering_agent", "tools", "lookup_track"],
        ["question_answering_agent", "lookup_track"],
    ) == 1.0


def test_trajectory_subsequence_rejects_missing_expected_step() -> None:
    assert trajectory_subsequence(["refund_agent"], ["refund_agent", "refund"]) == 0.5


def test_route_is_correct_compares_exact_route() -> None:
    assert route_is_correct("refund_agent", "refund_agent") is True
    assert route_is_correct("refund_agent", "question_answering_agent") is False
```

- [ ] **Step 2: Run the evaluator tests to verify they fail**

Run: `cd agent_sample/langchain-agent-evaluation && pytest tests/test_evaluation.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation'`.

- [ ] **Step 3: Implement local evaluation APIs**

Implement `trajectory_subsequence(actual: list[str], expected: list[str]) -> float` as an ordered-subsequence ratio over the expected list. Implement `route_is_correct(actual: str, expected: str) -> bool`. Add tutorial-derived examples for James Brown lookup, incomplete Aaron Mitchell refund, Led Zeppelin purchase lookup, Pink Floyd album lookup, and invoice 237 refund.

Implement:

```python
async def run_graph(graph: CompiledStateGraph, question: str) -> dict[str, object]:
    return await graph.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"env": "test"}},
    )
```

Add `run_with_trajectory`, `run_intent_classifier`, and `run_evaluation_suite` with the listed signatures and return types. Both graph runners must supply `config={"configurable": {"env": "test"}}`. The trajectory runner must stream debug events with subgraphs enabled, append entered node names, and append `tool_calls[*]["name"]` when the entered node is `tools`. The final-response judge must request a Pydantic Boolean/reasoning result from `qwen3:1.7b` and return both fields.

- [ ] **Step 4: Run the evaluator tests to verify they pass**

Run: `cd agent_sample/langchain-agent-evaluation && pytest tests/test_evaluation.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the local evaluators**

```bash
git add agent_sample/langchain-agent-evaluation/evaluation.py agent_sample/langchain-agent-evaluation/tests/test_evaluation.py
git commit -m "feat: add local agent evaluation suite"
```

### Task 7: Wire CLI behavior, documentation, and optional tracing

**Files:**
- Modify: `agent_sample/langchain-agent-evaluation/main.py`
- Modify: `agent_sample/langchain-agent-evaluation/README.md`
- Modify: `agent_sample/langchain-agent-evaluation/tests/test_main.py`

- [ ] **Step 1: Add failing CLI dispatch tests**

Use `monkeypatch` to prevent real model calls:

```python
import main


def test_main_runs_single_question_in_test_environment(monkeypatch) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(main, "create_support_graph", lambda database: object())
    monkeypatch.setattr(main, "ensure_database", lambda path: path)
    monkeypatch.setattr(
        main, "run_single_question",
        lambda graph, question: observed.update(graph=graph, question=question) or "answer",
    )
    monkeypatch.setattr("sys.argv", ["main.py", "--question", "albums by Prince"])

    main.main()

    assert observed["question"] == "albums by Prince"
```

- [ ] **Step 2: Run the CLI tests to verify they fail**

Run: `cd agent_sample/langchain-agent-evaluation && pytest tests/test_main.py -v`

Expected: FAIL because `main` does not yet wire the graph runner.

- [ ] **Step 3: Implement explicit CLI dispatch and errors**

Extend `main.py` so it:

1. Calls `ensure_database(Path(args.database))`.
2. Creates the parent graph once.
3. Runs `run_single_question` for `--question`, prints its normalized response, and runs `asyncio.run(run_evaluation_suite(graph, create_model()))` for `--evaluate`.
4. Catches only `requests.RequestException`, `OSError`, and Ollama connection exceptions, then exits with `parser.error()` including `ollama serve` and `ollama pull qwen3:1.7b` guidance for model errors.
5. Rejects the combination of no command and neither `--question` nor `--evaluate`; allow both explicit flags in one invocation, executing the question first.

Add `--langsmith-tracing` as an opt-in flag that sets `LANGSMITH_TRACING=true` only when `LANGSMITH_API_KEY` already exists; otherwise fail with an explicit missing-key error. Do not instantiate `langsmith.Client` or create cloud datasets.

- [ ] **Step 4: Complete the README**

Document:

- the Chinook first-run download and its public source;
- normal safe behavior (`env="test"` and no database deletion);
- the exact `--question` and `--evaluate` commands;
- expected Ollama prerequisites and troubleshooting;
- the optional `LANGSMITH_API_KEY` / `--langsmith-tracing` route, clarifying that inference remains Ollama-based;
- unit-test and local smoke-test commands.

- [ ] **Step 5: Run unit tests**

Run: `cd agent_sample/langchain-agent-evaluation && pytest tests -v`

Expected: PASS with no network or Ollama calls.

- [ ] **Step 6: Commit the CLI and documentation**

```bash
git add agent_sample/langchain-agent-evaluation/main.py agent_sample/langchain-agent-evaluation/README.md agent_sample/langchain-agent-evaluation/tests/test_main.py
git commit -m "feat: add evaluation CLI and documentation"
```

### Task 8: Validate against the local Ollama runtime

**Files:**
- Modify: `agent_sample/langchain-agent-evaluation/README.md` only if command output reveals an instruction mismatch.

- [ ] **Step 1: Install the declared dependencies**

Run:

```bash
cd agent_sample/langchain-agent-evaluation
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Expected: dependency installation succeeds.

- [ ] **Step 2: Run the full deterministic test suite**

Run: `cd agent_sample/langchain-agent-evaluation && .venv/bin/python -m pytest tests -v`

Expected: PASS.

- [ ] **Step 3: Verify the local model is available**

Run: `ollama list`

Expected: output includes `qwen3:1.7b`. If absent, run `ollama pull qwen3:1.7b` before continuing.

- [ ] **Step 4: Run a catalog smoke test**

Run:

```bash
cd agent_sample/langchain-agent-evaluation
.venv/bin/python main.py --question "What James Brown songs do you have?"
```

Expected: a readable catalog response from the local Ollama model and no mutation of `chinook.db`.

- [ ] **Step 5: Run the complete local evaluation suite**

Run:

```bash
cd agent_sample/langchain-agent-evaluation
.venv/bin/python main.py --evaluate
```

Expected: final-response, trajectory, and route results printed for every tutorial-derived example; the process does not request any cloud credential.

- [ ] **Step 6: Commit documentation corrections only if needed**

```bash
git add agent_sample/langchain-agent-evaluation/README.md
git commit -m "docs: clarify local evaluation validation"
```
