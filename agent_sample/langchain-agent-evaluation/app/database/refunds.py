import sqlite3
from pathlib import Path


def refund(
    database: Path,
    invoice_id: int | None,
    invoice_line_ids: list[int] | None,
    *,
    mock: bool,
) -> float:
    selected_invoice_line_ids = list(invoice_line_ids or [])
    if invoice_id is None and not selected_invoice_line_ids:
        return 0.0
    connection = sqlite3.connect(database)
    try:
        total = _refund_total(connection, invoice_id, selected_invoice_line_ids)
        if not mock:
            _delete_refunded_rows(connection, invoice_id, selected_invoice_line_ids)
            connection.commit()
        return total
    finally:
        connection.close()


def _refund_total(connection: sqlite3.Connection, invoice_id: int | None, invoice_line_ids: list[int]) -> float:
    total = 0.0
    if invoice_id is not None:
        row = connection.execute(
            "SELECT COALESCE(Total, 0.0) FROM Invoice WHERE InvoiceId = ?", (invoice_id,)
        ).fetchone()
        total += float((row or (0.0,))[0] or 0.0)
    if invoice_line_ids:
        placeholders = ", ".join("?" for _ in invoice_line_ids)
        row = connection.execute(
            f"SELECT COALESCE(SUM(UnitPrice * Quantity), 0.0) FROM InvoiceLine WHERE InvoiceLineId IN ({placeholders})",
            tuple(invoice_line_ids),
        ).fetchone()
        total += float((row or (0.0,))[0] or 0.0)
    return total


def _delete_refunded_rows(
    connection: sqlite3.Connection, invoice_id: int | None, invoice_line_ids: list[int]
) -> None:
    if invoice_line_ids:
        placeholders = ", ".join("?" for _ in invoice_line_ids)
        connection.execute(
            f"DELETE FROM InvoiceLine WHERE InvoiceLineId IN ({placeholders})", tuple(invoice_line_ids)
        )
    if invoice_id is not None:
        connection.execute("DELETE FROM InvoiceLine WHERE InvoiceId = ?", (invoice_id,))
        connection.execute("DELETE FROM Invoice WHERE InvoiceId = ?", (invoice_id,))
