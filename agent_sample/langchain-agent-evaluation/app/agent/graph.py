from pathlib import Path
from functools import partial

from langchain.agents import create_agent
from langgraph.graph import END, START, StateGraph

from app.schemas import AgentState, UserIntent
from app.refund import create_refund_graph
from app.tools import create_catalog_tools
from .model import create_model
from .components.compile_followup import compile_followup
from .components.intent_classifier import intent_classifier
from .components.question_answering_agent import question_answering_agent
from .components.refund_agent import refund_agent
from .question_answering import QUESTION_ANSWERING_SYSTEM_PROMPT


def create_support_graph(
    database: Path,
    model_factory=create_model,
    qa_factory=create_agent,
    catalog_tools_factory=create_catalog_tools,
    refund_factory=create_refund_graph,
):
    database_path = Path(database)
    model = model_factory()
    classifier_model = model.with_structured_output(UserIntent)
    question_answering_graph = qa_factory(
        model,
        tools=catalog_tools_factory(database_path),
        system_prompt=QUESTION_ANSWERING_SYSTEM_PROMPT,
        name="question_answering_agent",
    )
    refund_graph = refund_factory(model, database_path)

    graph = StateGraph(AgentState)
    graph.add_node(
        "intent_classifier",
        partial(intent_classifier, classifier_model),
        destinations=("refund_agent", "question_answering_agent"),
    )
    graph.add_node("question_answering_agent", partial(question_answering_agent, question_answering_graph))
    graph.add_node("refund_agent", partial(refund_agent, refund_graph))
    graph.add_node("compile_followup", compile_followup)
    graph.add_edge(START, "intent_classifier")
    graph.add_edge("question_answering_agent", "compile_followup")
    graph.add_edge("refund_agent", "compile_followup")
    graph.add_edge("compile_followup", END)
    return graph.compile(name="support_graph")