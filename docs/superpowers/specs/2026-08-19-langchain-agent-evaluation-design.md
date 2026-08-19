# LangChain Agent Evaluation with Ollama Design

## Goal

Create a runnable Python application in `agent_sample/langchain-agent-evaluation` based on LangSmith's "Evaluate a complex agent" tutorial. The application implements a digital music-store customer-support agent and evaluates its final response, execution trajectory, and intent-routing step.

All language-model inference uses a locally running Ollama model, `qwen3:1.7b`. The application must run its core evaluation workflow without cloud API credentials.

## Scope

The agent supports two request types:

- Music catalog lookup, such as finding tracks, artists, or albums.
- Refund requests, using a customer's purchase details or an invoice identifier.

It uses a parent LangGraph to classify the request and route it to one of two subgraphs:

1. A lookup ReAct agent with SQLite-backed track, artist, and album tools.
2. A refund graph that extracts purchase information, looks up matching purchases, requests missing identifying details, or simulates a refund.

The Chinook SQLite database is downloaded from the tutorial's public source on first use and stored inside the sample directory. The source database is never modified by the standard demo or evaluation commands: refund execution uses mock mode unless an explicitly named production mode is added in the future.

## Architecture

The sample is organized by responsibility:

- `main.py`: CLI entry point for a one-shot support request and the evaluation suite.
- `database.py`: Chinook download/initialization and parameterized SQLite queries.
- `tools.py`: LangChain tools for catalog lookup.
- `refund.py`: purchase-information extraction, purchase lookup, and mock-safe refund graph.
- `agent.py`: Ollama model construction, intent routing, lookup agent, and parent graph.
- `evaluation.py`: examples, local evaluators, trajectory capture, and formatted results.
- `schemas.py`: typed state and structured-output schemas shared by graph nodes.
- `tests/`: deterministic tests for database and local evaluation logic.
- `README.md`: requirements, setup, commands, safety behavior, and optional LangSmith configuration.

Small, testable functions keep external I/O (Ollama and database download) at module boundaries. The graph returns a normalized `followup` field so the CLI and evaluators share one stable response interface.

## Model Integration and Structured Output

`ChatOllama(model="qwen3:1.7b")` is used for the intent router, refund-information extraction, lookup agent, and final-answer judge. Prompt instructions require JSON matching explicit schemas for routing and extraction. Parsed values are validated before graph routing; malformed or incomplete model output produces a user-facing request for required information rather than an unsafe refund or fabricated lookup.

The local final-answer evaluator uses the same Ollama model as an LLM-as-judge, with a factual-equivalence prompt. Because local model evaluation is probabilistic, its reasoning and Boolean verdict are shown with each result. Deterministic trajectory and route evaluators complement it.

## Evaluation Design

The local suite contains tutorial-derived examples with expected final-answer facts, expected trajectory subsequences, and expected routes:

- **Final response:** invoke the full parent graph in mock mode and compare the normalized response to the reference through the local judge.
- **Trajectory:** stream parent and subgraph events, collect graph-node names and invoked lookup-tool names, then score expected steps as an ordered subsequence.
- **Single step:** invoke the intent-classifier node directly and compare its selected route to the expected route.

The CLI prints every case and a summary score. Network access is only needed once to obtain Chinook data; model inference uses the local Ollama server.

## Optional LangSmith Integration

`LANGSMITH_API_KEY` is never required. If the usual LangSmith environment variables are set, LangChain tracing can be enabled and the README documents how to send or inspect runs. The local evaluation implementation remains the default, so no LangSmith client, dataset creation, or cloud call blocks normal use.

## Error Handling and Safety

- Database download failures identify the URL and HTTP/status error and do not leave a partial database file.
- Missing database tables and invalid invoice identifiers surface clear command errors.
- All SQL values are passed as query parameters; dynamically sized `IN` clauses generate only placeholders.
- Evaluations and normal demonstrations set the graph configuration to `env="test"` to prevent deletion.
- Missing or unavailable Ollama is reported with a command-level error instructing the user to start Ollama and pull `qwen3:1.7b`.

## Validation

Unit tests avoid calling Ollama and cover database query/refund mock behavior, ordered-subsequence scoring, trajectory normalization, and exact route grading. The README's smoke-test command runs a representative catalog query against a local Ollama server, followed by the full local evaluation CLI. Optional LangSmith behavior is documented but is not part of the required test path.
