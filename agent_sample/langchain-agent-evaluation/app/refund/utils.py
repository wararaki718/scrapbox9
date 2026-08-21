from langchain_core.runnables import RunnableConfig

from app.schemas import PurchaseInformation
from .models import RefundState


def normalized_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def normalized_line_ids(value: list[int] | None) -> list[int] | None:
    if value is None:
        return None
    return [int(invoice_line_id) for invoice_line_id in value]


def merge_purchase_information(state: RefundState, purchase_information: PurchaseInformation) -> RefundState:
    merged_state: RefundState = dict(state)
    invoice_line_ids = normalized_line_ids(purchase_information.invoice_line_ids)
    merged_state["followup"] = normalized_text(purchase_information.followup)
    merged_state.update(_merge_value("invoice_id", purchase_information.invoice_id))
    merged_state.update(_merge_value("invoice_line_ids", invoice_line_ids))
    if purchase_information.invoice_id is not None:
        merged_state["invoice_line_ids"] = None
    elif invoice_line_ids:
        merged_state["invoice_id"] = None
    for key, value in (
        ("first_name", purchase_information.customer_first_name),
        ("last_name", purchase_information.customer_last_name),
        ("phone", purchase_information.customer_phone),
        ("track_name", purchase_information.track_name),
        ("album_title", purchase_information.album_title),
        ("artist_name", purchase_information.artist_name),
        ("purchase_date_iso_8601", purchase_information.purchase_date_iso_8601),
    ):
        merged_state.update(_merge_value(key, normalized_text(value)))
    return merged_state


def _merge_value(key: str, value: object) -> RefundState:
    return {key: value} if value is not None else {}


def configurable_value(config: RunnableConfig | None, key: str) -> object:
    if not config:
        return None
    configurable = config.get("configurable")
    return configurable.get(key) if isinstance(configurable, dict) else None
