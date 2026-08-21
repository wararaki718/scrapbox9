import sqlite3
from pathlib import Path

from .connection import fetch_rows
from .models import AlbumRow, ArtistRow, TrackRow


def find_tracks(database: Path, name: str | None, artist: str | None) -> list[TrackRow]:
    where_clauses: list[str] = []
    parameters: list[str] = []
    _append_optional_text_filter(where_clauses, parameters, "t.Name", name)
    _append_optional_text_filter(where_clauses, parameters, "ar.Name", artist)
    query = f"""
        SELECT t.Name AS track_name, al.Title AS album_title, ar.Name AS artist_name
        FROM Track t
        JOIN Album al ON al.AlbumId = t.AlbumId
        JOIN Artist ar ON ar.ArtistId = al.ArtistId
        {_where_sql(where_clauses)}
        ORDER BY lower(t.Name), lower(al.Title), lower(ar.Name), t.TrackId
        LIMIT 20
    """
    return [_track_row(row) for row in fetch_rows(database, query, parameters)]


def find_albums(database: Path, title: str | None, artist: str | None) -> list[AlbumRow]:
    where_clauses: list[str] = []
    parameters: list[str] = []
    _append_optional_text_filter(where_clauses, parameters, "al.Title", title)
    _append_optional_text_filter(where_clauses, parameters, "ar.Name", artist)
    query = f"""
        SELECT al.Title AS album_title, ar.Name AS artist_name
        FROM Album al
        JOIN Artist ar ON ar.ArtistId = al.ArtistId
        {_where_sql(where_clauses)}
        ORDER BY lower(al.Title), lower(ar.Name), al.AlbumId
        LIMIT 20
    """
    return [_album_row(row) for row in fetch_rows(database, query, parameters)]


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
    return [_artist_row(row) for row in fetch_rows(database, query, parameters)]


def _track_row(row: sqlite3.Row) -> TrackRow:
    return {"track_name": row["track_name"], "album_title": row["album_title"], "artist_name": row["artist_name"]}


def _album_row(row: sqlite3.Row) -> AlbumRow:
    return {"album_title": row["album_title"], "artist_name": row["artist_name"]}


def _artist_row(row: sqlite3.Row) -> ArtistRow:
    return {"artist_name": row["artist_name"]}


def _append_optional_text_filter(
    where_clauses: list[str], parameters: list[str], column: str, value: str | None
) -> None:
    if value is None:
        return
    where_clauses.append(f"lower({column}) = lower(?)")
    parameters.append(value)


def _where_sql(where_clauses: list[str]) -> str:
    return f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
