from pathlib import Path

from langchain_ollama import ChatOllama

from app.database import lookup_purchases
from app.database import refund as apply_refund

from .graph import create_refund_graph as _create_refund_graph
from .models import RefundState, RefundStep
from .prompts import MISSING_IDENTITY_FOLLOWUP, NO_PURCHASES_FOLLOWUP, REFUND_INSTRUCTIONS
from .router import next_refund_step
from .utils import merge_purchase_information, normalized_line_ids, normalized_text


def create_refund_graph(model: ChatOllama, database: Path):
    return _create_refund_graph(
        model,
        database,
        lookup_purchases=lookup_purchases,
        apply_refund=apply_refund,
    )


__all__ = [
    "MISSING_IDENTITY_FOLLOWUP",
    "NO_PURCHASES_FOLLOWUP",
    "REFUND_INSTRUCTIONS",
    "RefundState",
    "RefundStep",
    "create_refund_graph",
    "merge_purchase_information",
    "next_refund_step",
    "normalized_line_ids",
    "normalized_text",
]
