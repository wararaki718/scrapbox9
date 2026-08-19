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

## Safe entry points

These commands are intended to run with `env="test"` and must never delete Chinook purchase records:

```bash
python main.py --question "What James Brown songs do you have?"
python main.py --evaluate
```

This repository task only scaffolds the CLI and test harness. Later tasks will add the database, graph logic, and evaluation flow.
