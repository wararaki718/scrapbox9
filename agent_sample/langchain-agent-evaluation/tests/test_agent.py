import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent as agent_module
from schemas import UserIntent


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


class FakeQuestionAnsweringGraph:
    def __init__(self, response_message: AIMessage) -> None:
        self.response_message = response_message
        self.calls: list[tuple[dict[str, object], object]] = []

    def invoke(self, state: dict[str, object], config=None) -> dict[str, object]:
        self.calls.append((state, config))
        return {"messages": [self.response_message]}


class FakeRefundGraph:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.calls: list[tuple[dict[str, object], object]] = []

    def invoke(self, state: dict[str, object], config=None) -> dict[str, object]:
        self.calls.append((state, config))
        return self.result


def test_normalize_route_maps_supported_intents() -> None:
    assert agent_module.normalize_route("refund") == "refund_agent"
    assert agent_module.normalize_route("question_answering") == "question_answering_agent"


def test_create_model_uses_fixed_ollama_configuration() -> None:
    model = agent_module.create_model()

    assert model.model == "qwen3:1.7b"
    assert model.temperature == 0.0


def test_compile_followup_keeps_existing_followup() -> None:
    result = agent_module.compile_followup(
        {"followup": "Please confirm the invoice number.", "messages": [AIMessage(content="ignored")]}
    )

    assert result == {"followup": "Please confirm the invoice number."}


def test_compile_followup_extracts_text_from_message_content_blocks() -> None:
    result = agent_module.compile_followup(
        {
            "messages": [
                AIMessage(
                    content=[
                        {"type": "text", "text": "We have Black Dog available."},
                        {"type": "image_url", "image_url": "ignored"},
                        "Anything else I can help with?",
                    ]
                )
            ]
        }
    )

    assert result == {"followup": "We have Black Dog available.\nAnything else I can help with?"}


def test_create_support_graph_routes_questions_to_catalog_agent(tmp_path: Path, monkeypatch) -> None:
    fake_model = FakeModel("question_answering")
    create_model_calls: list[object] = []
    captured: dict[str, object] = {}
    qa_graph = FakeQuestionAnsweringGraph(
        AIMessage(content=[{"type": "text", "text": "Yes, we have Black Dog."}])
    )
    refund_graph = FakeRefundGraph({"followup": "unused"})

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
    assert len(qa_state["messages"]) == 1
    assert qa_state["messages"][0].content == "Do you have Black Dog?"
    assert isinstance(qa_config, dict)
    assert result["route"] == "question_answering_agent"
    assert result["followup"] == "Yes, we have Black Dog."


def test_create_support_graph_routes_refunds_into_refund_graph(tmp_path: Path, monkeypatch) -> None:
    fake_model = FakeModel("refund")
    captured: dict[str, object] = {}
    refund_graph = FakeRefundGraph(
        {
            "followup": "Previewed a refund total of $0.99.",
            "first_name": "Aaron",
            "last_name": "Mitchell",
            "phone": "+1 204",
            "invoice_line_ids": [6],
        }
    )

    def fake_create_model():
        return fake_model

    qa_graph = FakeQuestionAnsweringGraph(AIMessage(content="unused"))

    def fake_create_agent(*args, **kwargs):
        return qa_graph

    def fake_create_refund_graph(model, database: Path):
        captured["refund_model"] = model
        captured["refund_database"] = database
        return refund_graph

    monkeypatch.setattr(agent_module, "create_model", fake_create_model)
    monkeypatch.setattr(agent_module, "create_catalog_tools", lambda database: ())
    monkeypatch.setattr(agent_module, "create_agent", fake_create_agent)
    monkeypatch.setattr(agent_module, "create_refund_graph", fake_create_refund_graph)

    graph = agent_module.create_support_graph(tmp_path / "catalog.db")
    config = {"configurable": {"env": "test"}}

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Please refund Black Dog.")],
            "customer_first_name": "Aaron",
            "customer_last_name": "Mitchell",
            "customer_phone": "+1 204",
        },
        config=config,
    )

    child_state, child_config = refund_graph.calls[0]
    assert len(child_state["messages"]) == 1
    assert child_state["messages"][0].content == "Please refund Black Dog."
    assert child_state["followup"] is None
    assert child_state["invoice_id"] is None
    assert child_state["invoice_line_ids"] is None
    assert child_state["first_name"] == "Aaron"
    assert child_state["last_name"] == "Mitchell"
    assert child_state["phone"] == "+1 204"
    assert child_state["track_name"] is None
    assert child_state["album_title"] is None
    assert child_state["artist_name"] is None
    assert child_state["purchase_date_iso_8601"] is None
    assert isinstance(child_config, dict)
    assert child_config["configurable"]["env"] == "test"
    assert captured["refund_model"] is fake_model
    assert captured["refund_database"] == tmp_path / "catalog.db"
    assert qa_graph.calls == []
    assert result["route"] == "refund_agent"
    assert result["followup"] == "Previewed a refund total of $0.99."
    assert result["customer_first_name"] == "Aaron"
    assert result["customer_last_name"] == "Mitchell"
    assert result["customer_phone"] == "+1 204"
    assert result["invoice_line_ids"] == [6]
