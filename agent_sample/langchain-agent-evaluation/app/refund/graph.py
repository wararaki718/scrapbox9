from pathlib import Path
from functools import partial

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from .components.gather_info import gather_info
from .components.lookup import lookup
from .components.refund import refund
from .components.respond import respond
from .models import RefundState


def create_refund_graph(
    model: ChatOllama,
    database: Path,
    lookup_purchases,
    apply_refund,
):
    database_path = Path(database)
    graph = StateGraph(RefundState)
    graph.add_node("gather_info", partial(gather_info, model), destinations=("refund", "lookup", "respond"))
    graph.add_node("lookup", partial(lookup, database_path, lookup_purchases=lookup_purchases))
    graph.add_node("refund", partial(refund, database_path, apply_refund=apply_refund))
    graph.add_node("respond", respond)
    graph.add_edge(START, "gather_info")
    graph.add_edge("lookup", END)
    graph.add_edge("refund", END)
    graph.add_edge("respond", END)
    return graph.compile(name="refund_graph")
