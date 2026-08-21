from pathlib import Path

from langchain.agents import create_agent

from refund import create_refund_graph
from tools import create_catalog_tools

from .graph import create_support_graph as _create_support_graph
from .model import create_model
from .components.compile_followup import compile_followup
from .refund_adapter import parent_to_refund_state as _parent_to_refund_state
from .refund_adapter import refund_to_parent_update as _refund_to_parent_update
from .router import normalize_route
from .text import (
    content_block_to_text as _content_block_to_text,
    message_content_to_text as _message_content_to_text,
    normalized_text as _normalized_text,
)


def create_support_graph(database: Path):
    return _create_support_graph(
        database,
        model_factory=create_model,
        qa_factory=create_agent,
        catalog_tools_factory=create_catalog_tools,
        refund_factory=create_refund_graph,
    )


__all__ = ["create_model", "create_support_graph", "normalize_route"]