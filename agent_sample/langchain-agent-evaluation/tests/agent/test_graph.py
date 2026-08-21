import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import app.agent as agent_module
from app.schemas import UserIntent


class FakeStructuredRunnable:
    def __init__(self, result: UserIntent) -> None:
        self.result = result
        self.calls: list[list[object]] = []

    def invoke(self, messages: list[object]) -> UserIntent:
        self.calls.append(messages)
        return self.result


class FakeModel:
    def __init__(self, intent: str) -> None:
        self.intent = intent
        self.schemas: list[object] = []
        self.runnables: list[FakeStructuredRunnable] = []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        runnable = FakeStructuredRunnable(UserIntent(intent=self.intent))
        self.runnables.append(runnable)
        return runnable


class FakeGraph:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.calls: list[tuple[dict[str, object], object]] = []

    def invoke(self, state: dict[str, object], config=None) -> dict[str, object]:
        self.calls.append((state, config))
        return self.result


def test_create_support_graph_routes_questions_to_catalog_agent(tmp_path: Path, monkeypatch) -> None:
    fake_model = FakeModel("question_answering")
    create_model_calls: list[object] = []
    captured: dict[str, object] = {}
    qa_graph = FakeGraph({"messages": [AIMessage(content="Yes, we have Black Dog.")]})
    refund_graph = FakeGraph({"followup": "unused"})

    def fake_create_model():
        create_model_calls.append(object())
        return fake_model

    def fake_create_catalog_tools(database: Path):
        captured["catalog_database"] = database
        return ("lookup_track", "lookup_album", "lookup_artist")

    def fake_create_agent(model, tools, **kwargs):
        captured["qa_model"] = model
        captured["qa_tools"] = tools
        captured["qa_system_prompt"] = kwargs.get("system_prompt")
        return qa_graph

    def fake_create_refund_graph(model, database: Path):
        captured["refund_model"] = model
        captured["refund_database"] = database
        return refund_graph

    monkeypatch.setattr(agent_module, "create_model", fake_create_model)
    monkeypatch.setattr(agent_module, "create_catalog_tools", fake_create_catalog_tools)
    monkeypatch.setattr(agent_module, "create_agent", fake_create_agent)
    monkeypatch.setattr(agent_module, "create_refund_graph", fake_create_refund_graph)

    graph = agent_module.create_support_graph(tmp_path / "catalog.db")
    result = graph.invoke({"messages": [HumanMessage(content="Do you have Black Dog?")]})

    assert len(create_model_calls) == 1
    assert fake_model.schemas == [UserIntent]
    classifier_messages = fake_model.runnables[0].calls[0]
    assert isinstance(classifier_messages[0], SystemMessage)
    assert "music store" in classifier_messages[0].content
    assert "refund" in classifier_messages[0].content
    assert "question_answering" in classifier_messages[0].content
    assert captured["catalog_database"] == tmp_path / "catalog.db"
    assert captured["qa_model"] is fake_model
    assert captured["qa_tools"] == ("lookup_track", "lookup_album", "lookup_artist")
    assert captured["refund_model"] is fake_model
    assert captured["refund_database"] == tmp_path / "catalog.db"
    qa_state, qa_config = qa_graph.calls[0]
    assert qa_state["messages"][0].content == "Do you have Black Dog?"
    assert isinstance(qa_config, dict)
    assert result["route"] == "question_answering_agent"
    assert result["followup"] == "Yes, we have Black Dog."


def test_create_support_graph_routes_refunds_into_refund_graph(tmp_path: Path, monkeypatch) -> None:
    fake_model = FakeModel("refund")
    refund_graph = FakeGraph(
        {
            "followup": "Previewed a refund total of $0.99.",
            "first_name": "Aaron",
            "last_name": "Mitchell",
            "phone": "+1 204",
            "invoice_line_ids": [6],
        }
    )

    monkeypatch.setattr(agent_module, "create_model", lambda: fake_model)
    monkeypatch.setattr(agent_module, "create_catalog_tools", lambda database: ())
    monkeypatch.setattr(agent_module, "create_agent", lambda *args, **kwargs: FakeGraph({"messages": []}))
    monkeypatch.setattr(agent_module, "create_refund_graph", lambda model, database: refund_graph)

    graph = agent_module.create_support_graph(tmp_path / "catalog.db")
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Please refund Black Dog.")],
            "customer_first_name": "Aaron",
            "customer_last_name": "Mitchell",
            "customer_phone": "+1 204",
        },
        config={"configurable": {"env": "test"}},
    )

    child_state, child_config = refund_graph.calls[0]
    assert child_state["messages"][0].content == "Please refund Black Dog."
    assert child_state["first_name"] == "Aaron"
    assert child_state["last_name"] == "Mitchell"
    assert child_state["phone"] == "+1 204"
    assert child_config["configurable"]["env"] == "test"
    assert result["route"] == "refund_agent"
    assert result["followup"] == "Previewed a refund total of $0.99."
    assert result["customer_first_name"] == "Aaron"
    assert result["invoice_line_ids"] == [6]
