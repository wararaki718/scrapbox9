from collections.abc import Mapping, Sequence

from langgraph.graph.state import CompiledStateGraph

from .response import normalize_result

TEST_CONFIG = {"configurable": {"env": "test"}}


async def run_graph(graph: CompiledStateGraph, question: str) -> dict[str, object]:
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config=TEST_CONFIG,
    )
    return normalize_result(result)


async def run_intent_classifier(graph: CompiledStateGraph, messages: Sequence[object]) -> str:
    command = await graph.nodes["intent_classifier"].ainvoke(
        {"messages": list(messages)},
        config=TEST_CONFIG,
    )
    goto = getattr(command, "goto", None)
    if isinstance(goto, str):
        return goto
    if isinstance(command, Mapping):
        mapped_goto = command.get("goto")
        if isinstance(mapped_goto, str):
            return mapped_goto
    raise ValueError("intent_classifier did not return a route")
