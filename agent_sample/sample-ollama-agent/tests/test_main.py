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

    def test_main_reports_ollama_help_when_model_invocation_fails(self) -> None:
        class FailingModel:
            def invoke(self, prompt: str) -> FakeResponse:
                raise ConnectionError("connection refused")

        output = io.StringIO()

        with patch.object(main, "create_chat_model", return_value=FailingModel()), redirect_stdout(output):
            exit_code = main.main(["hello"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            output.getvalue(),
            "Could not get a response from Ollama: connection refused\n"
            "Start Ollama with: ollama serve\n"
            "Download the model with: ollama pull qwen2.5:0.5b\n",
        )

    def test_main_reports_ollama_help_when_model_creation_fails(self) -> None:
        output = io.StringIO()

        with (
            patch.object(main, "create_chat_model", side_effect=ModuleNotFoundError("langchain_ollama")),
            redirect_stdout(output),
        ):
            exit_code = main.main(["hello"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            output.getvalue(),
            "Could not get a response from Ollama: langchain_ollama\n"
            "Start Ollama with: ollama serve\n"
            "Download the model with: ollama pull qwen2.5:0.5b\n",
        )

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

    def test_interactive_mode_reports_ollama_help_when_model_creation_fails(self) -> None:
        output: list[str] = []

        exit_code = main.run_interactive(
            output=output.append,
            model_factory=lambda: (_ for _ in ()).throw(ConnectionError("connection refused")),
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            output,
            [
                "Could not get a response from Ollama: connection refused",
                "Start Ollama with: ollama serve",
                "Download the model with: ollama pull qwen2.5:0.5b",
            ],
        )

    def test_interactive_mode_handles_eof(self) -> None:
        output: list[str] = []

        exit_code = main.run_interactive(
            input_fn=lambda _: (_ for _ in ()).throw(EOFError),
            output=output.append,
            model_factory=FakeModel,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(output, ["Goodbye."])

    def test_interactive_mode_handles_keyboard_interrupt(self) -> None:
        output: list[str] = []

        exit_code = main.run_interactive(
            input_fn=lambda _: (_ for _ in ()).throw(KeyboardInterrupt),
            output=output.append,
            model_factory=FakeModel,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(output, ["Goodbye."])

    def test_interactive_mode_recovers_from_model_invocation_failure(self) -> None:
        class FailingModel:
            def invoke(self, prompt: str) -> FakeResponse:
                raise ConnectionError("connection refused")

        answers = iter(["bad", "exit"])
        output: list[str] = []

        exit_code = main.run_interactive(
            input_fn=lambda _: next(answers),
            output=output.append,
            model_factory=FailingModel,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            output.count("Could not get a response from Ollama: connection refused"),
            1,
        )
        self.assertEqual(output.count("Start Ollama with: ollama serve"), 1)
        self.assertEqual(
            output.count("Download the model with: ollama pull qwen2.5:0.5b"),
            1,
        )
        self.assertEqual(output[-1], "Goodbye.")


if __name__ == "__main__":
    unittest.main()