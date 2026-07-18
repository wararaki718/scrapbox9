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

	try:
		model = create_chat_model()
		return 0 if run_prompt(model, " ".join(arguments), print) else 1
	except Exception as error:
		print_ollama_help(error, print)
		return 1


def run_interactive(
	input_fn: Callable[[str], str] = input,
	output: Callable[[str], None] = print,
	model_factory: Callable[[], Any] = create_chat_model,
) -> int:
	try:
		model = model_factory()
	except Exception as error:
		print_ollama_help(error, output)
		return 1

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


if __name__ == "__main__":
	raise SystemExit(main())
