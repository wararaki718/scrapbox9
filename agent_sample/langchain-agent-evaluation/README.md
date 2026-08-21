# LangChain Agent Evaluation with Ollama

## Setup

```bash
cd agent_sample/langchain-agent-evaluation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama serve
ollama pull qwen3:1.7b
```

This sample always keeps inference on your local Ollama runtime. The default model is `qwen3:1.7b`.

## Chinook database

The first CLI run downloads `chinook.db` automatically when the file is missing:

```bash
python -m app.main --question "What James Brown songs do you have?"
```

- Download target: `./chinook.db` by default, or `--database <path>` if provided.
- Public download URL used by the sample: `https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db`
- Official Chinook project source: `https://github.com/lerocha/chinook-database`

## Safe local commands

These entry points run the graph with `env="test"`, so refund flows stay in preview mode and do **not** delete Chinook purchase rows.

Ask one question:

```bash
python -m app.main --question "What James Brown songs do you have?"
```

Run the evaluation suite:

```bash
python -m app.main --evaluate
```

Run both in one invocation (question first, then evaluations):

```bash
python -m app.main --question "Refund invoice 237." --evaluate
```

If you run `python -m app.main` with no action flags, the CLI prints help and exits successfully.

## Optional LangSmith tracing

Tracing is opt-in and only enabled when you already have a LangSmith API key:

```bash
export LANGSMITH_API_KEY=your_key_here
python -m app.main --question "What James Brown songs do you have?" --langsmith-tracing
```

- `--langsmith-tracing` sets `LANGSMITH_TRACING=true`
- if `LANGSMITH_API_KEY` is missing, the CLI stops with a parser error
- tracing is optional; model inference still stays local through Ollama
- this sample does **not** create LangSmith datasets or call `langsmith.Client`

## Ollama troubleshooting

If the CLI cannot reach Ollama or the model is unavailable:

1. Start the local server:

   ```bash
   ollama serve
   ```

2. Download the required model:

   ```bash
   ollama pull qwen3:1.7b
   ```

3. Re-run one of the smoke tests:

   ```bash
   python -m app.main --question "What James Brown songs do you have?"
   python -m app.main --evaluate
   ```

## Test commands

Run the focused CLI tests:

```bash
pytest tests/test_main.py -v
```

Run the full sample test suite:

```bash
pytest tests -v
```
