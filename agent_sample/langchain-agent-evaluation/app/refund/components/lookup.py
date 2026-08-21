from pathlib import Path

from tabulate import tabulate

from ..models import RefundState
from ..prompts import NO_PURCHASES_FOLLOWUP


def lookup(database: Path, state: RefundState, lookup_purchases):
    rows = lookup_purchases(
        database,
        state.get("first_name") or "",
        state.get("last_name") or "",
        state.get("phone") or "",
        state.get("track_name"),
        state.get("album_title"),
        state.get("artist_name"),
        state.get("purchase_date_iso_8601"),
    )
    if not rows:
        return {"invoice_line_ids": [], "followup": NO_PURCHASES_FOLLOWUP}
    return {
        "invoice_line_ids": [row["invoice_line_id"] for row in rows],
        "followup": "I found these matching purchases:\n" + tabulate(rows, headers="keys", tablefmt="github", floatfmt=".2f"),
    }
