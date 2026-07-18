# Local LangChain Ollama Sample Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small Python CLI that uses LangChain and a local Ollama `qwen2.5:0.5b` model for one-shot and interactive questions.

**Architecture:** Keep the application in one focused module, with model creation, one-shot execution, interactive execution, and CLI dispatch as separate functions. Pass the model factory and input/output functions into behavior functions so tests use a fake model and never contact Ollama.

**Tech Stack:** Python 3.11, LangChain, `langchain-ollama`, Ollama, standard-library `unittest`.

---

## File Structure

- `agent_sample/sample-ollama-agent/main.py`: CLI behavior and LangChain Ollama integration.
- `agent_sample/sample-ollama-agent/requirements.txt`: runtime dependency pin range.
- `agent_sample/sample-ollama-agent/tests/test_main.py`: isolated unit tests using a fake chat model.
- `agent_sample/sample-ollama-agent/README.md`: setup, Ollama model download, and usage instructions.

### Task 1: Establish the Test Harness and One-Shot Contract

**Files:**
- Create: `agent_sample/sample-ollama-agent/tests/test_main.py`
- Create: `agent_sample/sample-ollama-agent/requirements.txt`

- [ ] **Step 1: Write the failing one-shot tests**

```python
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    def __init__(self, response: str = "sample answer") -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return FakeResponse(self.response)


class MainTests(unittest.TestCase):
    def test_default_model_is_small_qwen_model(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(main.get_model_name(), "qwen2.5:0.5b")

    def test_environment_variable_overrides_model(self) -> None:
        with patch.dict("os.environ", {"OLLAMA_MODEL": "tinyllama"}):
            self.assertEqual(main.get_model_name(), "tinyllama")

    def test_main_sends_joined_arguments_to_model(self) -> None:
        model = FakeModel("Tokyo")
        output = io.StringIO()

        with patch.object(main, "create_chat_model", return_value=model), redirect_stdout(output):
            exit_code = main.main(["Japan", "capital"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(model.prompts, ["Japan capital"])
        self.assertEqual(output.getvalue(), "Tokyo\n")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd agent_sample/sample-ollama-agent && python3 -m unittest discover -s tests -v`

Expected: FAIL because `main` does not define `get_model_name`, `create_chat_model`, or `main`.

- [ ] **Step 3: Add the runtime dependency declaration**

Create `agent_sample/sample-ollama-agent/requirements.txt`:

```text
langchain-ollama>=0.3,<1
```

- [ ] **Step 4: Implement the minimum one-shot behavior**

Replace `agent_sample/sample-ollama-agent/main.py` with:

```python
import os
import sys
from collections.abc import Callable, Sequence
from typing import Any


DEFAULT_MODEL = "qwen2.5:0.5b"


def get_model_name() -> str:
    return os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)


def create_chat_model() -> Any:
    from langchain_ollama import ChatOllama

    return ChatOllama(model=get_model_name())


def ask(model: Any, prompt: str) -> str:
    response = model.invoke(prompt)
    return str(getattr(response, "content", response))


def print_ollama_help(error: Exception, output: Callable[[str], None]) -> None:
    output(f"Could not get a response from Ollama: {error}")
    output("Start Ollama with: ollama serve")
    output(f"Download the model with: ollama pull {get_model_name()}")


def run_prompt(model: Any, prompt: str, output: Callable[[str], None]) -> bool:
    try:
        output(ask(model, prompt))
    except Exception as error:
        print_ollama_help(error, output)
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        return run_interactive()

    model = create_chat_model()
    return 0 if run_prompt(model, " ".join(arguments), print) else 1


def run_interactive(
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
    model_factory: Callable[[], Any] = create_chat_model,
) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the one-shot tests to verify they pass**

Run: `cd agent_sample/sample-ollama-agent && python3 -m unittest discover -s tests -v`

Expected: PASS for the three one-shot tests.

### Task 2: Add Interactive Mode and Recoverable Error Handling

**Files:**
- Modify: `agent_sample/sample-ollama-agent/tests/test_main.py`
- Modify: `agent_sample/sample-ollama-agent/main.py`

- [ ] **Step 1: Write failing interactive-mode tests**

Add these methods to `MainTests`:

```python
    def test_interactive_mode_replies_then_exits(self) -> None:
        model = FakeModel("local answer")
        answers = iter(["hello", "exit"])
        output: list[str] = []

        exit_code = main.run_interactive(
            input_fn=lambda _: next(answers),
            output=output.append,
            model_factory=lambda: model,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(model.prompts, ["hello"])
        self.assertEqual(output, ["local answer", "Goodbye."])

    def test_interactive_mode_handles_eof(self) -> None:
        output: list[str] = []

        exit_code = main.run_interactive(
            input_fn=lambda _: (_ for _ in ()).throw(EOFError),
            output=output.append,
            model_factory=FakeModel,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(output, ["Goodbye."])

    def test_model_failure_shows_ollama_recovery_commands(self) -> None:
        class FailingModel:
            def invoke(self, prompt: str) -> FakeResponse:
                raise ConnectionError("connection refused")

        output: list[str] = []
        success = main.run_prompt(FailingModel(), "hello", output.append)

        self.assertFalse(success)
        self.assertIn("ollama serve", output[1])
        self.assertIn("ollama pull qwen2.5:0.5b", output[2])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd agent_sample/sample-ollama-agent && python3 -m unittest discover -s tests -v`

Expected: FAIL because `run_interactive` raises `NotImplementedError`.

- [ ] **Step 3: Implement the interactive loop**

Replace `run_interactive` in `agent_sample/sample-ollama-agent/main.py` with:

```python
def run_interactive(
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
    model_factory: Callable[[], Any] = create_chat_model,
) -> int:
    model = model_factory()

    while True:
        try:
            prompt = input_fn("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            output("Goodbye.")
            return 0

        if prompt.lower() in {"exit", "quit"}:
            output("Goodbye.")
            return 0
        if prompt:
            run_prompt(model, prompt, output)
```

- [ ] **Step 4: Run the full test suite to verify it passes**

Run: `cd agent_sample/sample-ollama-agent && python3 -m unittest discover -s tests -v`

Expected: PASS for all six tests.

### Task 3: Document Installation and Usage

**Files:**
- Modify: `agent_sample/sample-ollama-agent/README.md`

- [ ] **Step 1: Write the user-facing setup instructions**

Replace `agent_sample/sample-ollama-agent/README.md` with:

```markdown
# LangChain + Local Ollama Quickstart

This sample sends questions to a local Ollama model through LangChain. It uses
the small `qwen2.5:0.5b` model by default.

## Prerequisites

1. Install [Ollama](https://ollama.com/).
2. Download the default model:

   ```bash
   ollama pull qwen2.5:0.5b
   ```

   Ollama normally starts its local service automatically. If it is not
   running, start it with `ollama serve`.

## Setup

```bash
cd agent_sample/sample-ollama-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Send one question:

```bash
python main.py "日本の首都はどこですか？"
```

Or start an interactive prompt:

```bash
python main.py
```

Enter `exit` or `quit`, press Ctrl-D, or press Ctrl-C to finish interactive
mode.

## Choose Another Model

Set `OLLAMA_MODEL` before running the program:

```bash
OLLAMA_MODEL=tinyllama python main.py "Hello"
```

Download a selected model before use, for example:

```bash
ollama pull tinyllama
```

## Test

```bash
python3 -m unittest discover -s tests -v
```
```

- [ ] **Step 2: Run the tests after documentation changes**

Run: `cd agent_sample/sample-ollama-agent && python3 -m unittest discover -s tests -v`

Expected: PASS for all six tests.

- [ ] **Step 3: Inspect the working-tree changes**

Run: `git diff -- agent_sample/sample-ollama-agent docs/superpowers/plans/2026-07-18-local-langchain-ollama.md`

Expected: Only the intended sample implementation, test suite, dependency file, README, and plan changes are present.