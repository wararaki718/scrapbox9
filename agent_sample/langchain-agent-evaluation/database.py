from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TypedDict

import requests

DATABASE_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
DOWNLOAD_TIMEOUT = (5, 30)


class PurchaseLookupRow(TypedDict):
    invoice_line_id: int
    track_name: str
    artist_name: str
    purchase_date: str
    quantity_purchased: int
    price_per_unit: float


class TrackRow(TypedDict):
    track_name: str
    album_title: str
    artist_name: str


class AlbumRow(TypedDict):
    album_title: str
    artist_name: str


class ArtistRow(TypedDict):
    artist_name: str


def ensure_database(path: Path) -> Path:
    if path.is_file() and path.stat().st_size > 0:
        return path

    temporary_path = path.with_name(f".{path.name}.download")
    response = None
    try:
        response = requests.get(DATABASE_URL, timeout=DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()
        with temporary_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
        temporary_path.replace(path)
        return path
    except Exception as exc:  # pragma: no cover - exercised via tests
        if temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError(f"Failed to download Chinook database to {path}") from exc
    finally:
        if response is not None:
            close = getattr(response, "close", None)
            if callable(close):
                close()


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
    _append_optional_text_filter(where_clauses, parameters, "t.Name", track_name)
    _append_optional_text_filter(where_clauses, parameters, "al.Title", album_title)
    _append_optional_text_filter(where_clauses, parameters, "ar.Name", artist_name)
    if purchase_date_iso_8601 is not None:
        where_clauses.append("date(i.InvoiceDate) = date(?)")
        parameters.append(purchase_date_iso_8601)

    query = f"""
        SELECT
            il.InvoiceLineId AS invoice_line_id,
            t.Name AS track_name,
            ar.Name AS artist_name,
            substr(i.InvoiceDate, 1, 10) AS purchase_date,
            il.Quantity AS quantity_purchased,
            il.UnitPrice AS price_per_unit
        FROM Customer c
        JOIN Invoice i ON i.CustomerId = c.CustomerId
        JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
        JOIN Track t ON t.TrackId = il.TrackId
        JOIN Album al ON al.AlbumId = t.AlbumId
        JOIN Artist ar ON ar.ArtistId = al.ArtistId
        WHERE {" AND ".join(where_clauses)}
        ORDER BY substr(i.InvoiceDate, 1, 10), il.InvoiceLineId
    """
    return [_purchase_lookup_row(row) for row in _fetch_rows(database, query, parameters)]


def find_tracks(database: Path, name: str | None, artist: str | None) -> list[TrackRow]:
    where_clauses: list[str] = []
    parameters: list[str] = []
    _append_optional_text_filter(where_clauses, parameters, "t.Name", name)
    _append_optional_text_filter(where_clauses, parameters, "ar.Name", artist)

    query = f"""
        SELECT
            t.Name AS track_name,
            al.Title AS album_title,
            ar.Name AS artist_name
        FROM Track t
        JOIN Album al ON al.AlbumId = t.AlbumId
        JOIN Artist ar ON ar.ArtistId = al.ArtistId
        {_where_sql(where_clauses)}
        ORDER BY lower(t.Name), lower(al.Title), lower(ar.Name), t.TrackId
        LIMIT 20
    """
    return [_track_row(row) for row in _fetch_rows(database, query, parameters)]


def find_albums(database: Path, title: str | None, artist: str | None) -> list[AlbumRow]:
    where_clauses: list[str] = []
    parameters: list[str] = []
    _append_optional_text_filter(where_clauses, parameters, "al.Title", title)
    _append_optional_text_filter(where_clauses, parameters, "ar.Name", artist)

    query = f"""
        SELECT
            al.Title AS album_title,
            ar.Name AS artist_name
        FROM Album al
        JOIN Artist ar ON ar.ArtistId = al.ArtistId
        {_where_sql(where_clauses)}
        ORDER BY lower(al.Title), lower(ar.Name), al.AlbumId
        LIMIT 20
    """
    return [_album_row(row) for row in _fetch_rows(database, query, parameters)]


def find_artists(database: Path, name: str | None) -> list[ArtistRow]:
    where_clauses: list[str] = []
    parameters: list[str] = []
    _append_optional_text_filter(where_clauses, parameters, "ar.Name", name)

    query = f"""
        SELECT ar.Name AS artist_name
        FROM Artist ar
        {_where_sql(where_clauses)}
        ORDER BY lower(ar.Name), ar.ArtistId
        LIMIT 20
    """
    return [_artist_row(row) for row in _fetch_rows(database, query, parameters)]


def _purchase_lookup_row(row: sqlite3.Row) -> PurchaseLookupRow:
    return PurchaseLookupRow(
        invoice_line_id=row["invoice_line_id"],
        track_name=row["track_name"],
        artist_name=row["artist_name"],
        purchase_date=row["purchase_date"],
        quantity_purchased=row["quantity_purchased"],
        price_per_unit=row["price_per_unit"],
    )


def _track_row(row: sqlite3.Row) -> TrackRow:
    return TrackRow(
        track_name=row["track_name"],
        album_title=row["album_title"],
        artist_name=row["artist_name"],
    )


def _album_row(row: sqlite3.Row) -> AlbumRow:
    return AlbumRow(
        album_title=row["album_title"],
        artist_name=row["artist_name"],
    )


def _artist_row(row: sqlite3.Row) -> ArtistRow:
    return ArtistRow(
        artist_name=row["artist_name"],
    )


def _refund_total(
    connection: sqlite3.Connection,
    invoice_id: int | None,
    invoice_line_ids: list[int],
) -> float:
    total = 0.0
    if invoice_id is not None:
        row = connection.execute(
            "SELECT COALESCE(Total, 0.0) FROM Invoice WHERE InvoiceId = ?",
            (invoice_id,),
        ).fetchone()
        total += float((row or (0.0,))[0] or 0.0)

    if invoice_line_ids:
        placeholders = ", ".join("?" for _ in invoice_line_ids)
        row = connection.execute(
            f"""
            SELECT COALESCE(SUM(UnitPrice * Quantity), 0.0)
            FROM InvoiceLine
            WHERE InvoiceLineId IN ({placeholders})
            """,
            tuple(invoice_line_ids),
        ).fetchone()
        total += float((row or (0.0,))[0] or 0.0)

    return total


def _delete_refunded_rows(
    connection: sqlite3.Connection,
    invoice_id: int | None,
    invoice_line_ids: list[int],
) -> None:
    if invoice_line_ids:
        placeholders = ", ".join("?" for _ in invoice_line_ids)
        connection.execute(
            f"DELETE FROM InvoiceLine WHERE InvoiceLineId IN ({placeholders})",
            tuple(invoice_line_ids),
        )

    if invoice_id is not None:
        connection.execute("DELETE FROM InvoiceLine WHERE InvoiceId = ?", (invoice_id,))
        connection.execute("DELETE FROM Invoice WHERE InvoiceId = ?", (invoice_id,))


def _append_optional_text_filter(
    where_clauses: list[str],
    parameters: list[str],
    column: str,
    value: str | None,
) -> None:
    if value is None:
        return
    where_clauses.append(f"lower({column}) = lower(?)")
    parameters.append(value)


def _where_sql(where_clauses: list[str]) -> str:
    if not where_clauses:
        return ""
    return f"WHERE {' AND '.join(where_clauses)}"


def _fetch_rows(
    database: Path,
    query: str,
    parameters: list[str],
) -> list[sqlite3.Row]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(query, tuple(parameters)).fetchall()
    finally:
        connection.close()
