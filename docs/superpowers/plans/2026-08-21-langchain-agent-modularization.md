# LangChain Agent Modularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the evaluation support agent into focused Python modules while preserving the existing `import agent` API and behavior.

**Architecture:** The `agent/` package owns the implementation. Its `__init__.py` remains the compatibility boundary and exposes the existing public functions, while graph construction receives its model and child-graph factories as dependencies so existing monkeypatch-based tests continue to work. The existing `agent.py` remains as a legacy facade for direct file consumers.

**Tech Stack:** Python 3.11, LangChain, LangGraph, pytest.

---

### Task 1: Extract focused agent modules

**Files:**
- Create: `agent_sample/langchain-agent-evaluation/agent/model.py`
- Create: `agent_sample/langchain-agent-evaluation/agent/text.py`
- Create: `agent_sample/langchain-agent-evaluation/agent/refund_adapter.py`
- Create: `agent_sample/langchain-agent-evaluation/agent/question_answering.py`
- Create: `agent_sample/langchain-agent-evaluation/agent/router.py`
- Create: `agent_sample/langchain-agent-evaluation/agent/graph.py`
- Create: `agent_sample/langchain-agent-evaluation/agent/__init__.py`
- Modify: `agent_sample/langchain-agent-evaluation/agent.py`

- [ ] Move each responsibility without changing function behavior or signatures.
- [ ] Export the current public symbols from `agent/__init__.py`.
- [ ] Keep `create_support_graph` compatible with patched `create_model`, catalog tools, QA agent factory, and refund graph factory.
- [ ] Run `pytest agent_sample/langchain-agent-evaluation/tests/test_agent.py -q`.

### Task 2: Verify the full evaluation sample

**Files:**
- Test: `agent_sample/langchain-agent-evaluation/tests/`

- [ ] Run the complete sample test suite with `pytest agent_sample/langchain-agent-evaluation/tests -q`.
- [ ] Confirm only the requested agent modularization files changed.
