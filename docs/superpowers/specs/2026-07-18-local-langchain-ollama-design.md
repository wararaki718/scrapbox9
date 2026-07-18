# Local LangChain Ollama Sample Design

## Goal

Provide a small, standalone Python example that uses LangChain with a local
Ollama model. The default model is `qwen2.5:0.5b`, selected for its small size
and Japanese-language capability.

## Scope

The sample lives in `agent_sample/sample-ollama-agent/` and consists of:

- `main.py`, a command-line application.
- `README.md`, setup and usage documentation.
- Focused automated tests for command-line behavior that do not require an
  Ollama server or downloaded model.

## Behavior

- `python main.py "question"` sends one question and prints the response.
- `python main.py` starts an interactive prompt.
- In interactive mode, `exit`, `quit`, EOF, or Ctrl-C ends the program without
  a traceback.
- `OLLAMA_MODEL` overrides the default `qwen2.5:0.5b` model name.
- The application creates `ChatOllama` from `langchain-ollama` and invokes it
  with the user's text.
- Connection errors explain how to start Ollama and pull the selected model.

## Design

`main.py` separates model creation, single-prompt execution, interactive mode,
and command-line dispatch. This allows tests to supply a fake model callable
without connecting to Ollama. The application remains intentionally free of
agents, tools, memory, and streaming so the LangChain-to-local-model boundary
is clear.

## Dependencies and Setup

The README documents Python virtual-environment setup, installation of
`langchain-ollama`, installation of Ollama, `ollama pull qwen2.5:0.5b`, and
both command forms. It also documents the environment-variable override.

## Error Handling

Unexpected model invocation failures are caught at the command-line boundary.
The user receives a concise error plus the Ollama start and model-pull
commands; the program exits with a nonzero status for one-shot execution and
returns to the prompt for interactive execution.

## Testing

Tests use a fake chat model and verify command-line dispatch, interactive exit
handling, model-name selection, response rendering, and human-readable
connection guidance. They do not need a running Ollama service.