import sqlite3
from pathlib import Path

from .connection import fetch_rows
from .models import PurchaseLookupRow

LOOKUP_PURCHASES_LIMIT = 20


def lookup_purchases(
    database: Path,
    first_name: str,
    last_name: str,
    phone: str,
    track_name: str | None,
    album_title: str | None,
    artist_name: str | None,
    purchase_date_iso_8601: str | None,
) -> list[PurchaseLookupRow]:
    where_clauses = [
        "lower(c.FirstName) = lower(?)",
        "lower(c.LastName) = lower(?)",
        "lower(c.Phone) = lower(?)",
    ]
    parameters: list[str] = [first_name, last_name, phone]
    _append_filter(where_clauses, parameters, "t.Name", track_name)
    _append_filter(where_clauses, parameters, "al.Title", album_title)
    _append_filter(where_clauses, parameters, "ar.Name", artist_name)
    if purchase_date_iso_8601 is not None:
        where_clauses.append("date(i.InvoiceDate) = date(?)")
        parameters.append(purchase_date_iso_8601)
    query = f"""
        SELECT il.InvoiceLineId AS invoice_line_id, t.Name AS track_name,
               ar.Name AS artist_name, substr(i.InvoiceDate, 1, 10) AS purchase_date,
               il.Quantity AS quantity_purchased, il.UnitPrice AS price_per_unit
        FROM Customer c
        JOIN Invoice i ON i.CustomerId = c.CustomerId
        JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
        JOIN Track t ON t.TrackId = il.TrackId
        JOIN Album al ON al.AlbumId = t.AlbumId
        JOIN Artist ar ON ar.ArtistId = al.ArtistId
        WHERE {" AND ".join(where_clauses)}
        ORDER BY substr(i.InvoiceDate, 1, 10), il.InvoiceLineId
        LIMIT {LOOKUP_PURCHASES_LIMIT}
    """
    return [_purchase_row(row) for row in fetch_rows(database, query, parameters)]


def _purchase_row(row: sqlite3.Row) -> PurchaseLookupRow:
    return {
        "invoice_line_id": row["invoice_line_id"],
        "track_name": row["track_name"],
        "artist_name": row["artist_name"],
        "purchase_date": row["purchase_date"],
        "quantity_purchased": row["quantity_purchased"],
        "price_per_unit": row["price_per_unit"],
    }


def _append_filter(
    where_clauses: list[str], parameters: list[str], column: str, value: str | None
) -> None:
    if value is not None:
        where_clauses.append(f"lower({column}) = lower(?)")
        parameters.append(value)
