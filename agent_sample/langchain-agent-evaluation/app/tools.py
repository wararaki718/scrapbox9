from __future__ import annotations

import json
from pathlib import Path

from langchain.tools import tool

from app.database import find_albums, find_artists, find_tracks


def create_catalog_tools(database: Path):
    database_path = Path(database)

    @tool
    def lookup_track(name: str | None = None, artist: str | None = None) -> str:
        """Use when a user asks whether a specific track exists. Returns a JSON array of {track_name, album_title, artist_name} rows filtered by exact track name and optional exact artist."""

        return json.dumps(find_tracks(database_path, name, artist))

    @tool
    def lookup_album(title: str | None = None, artist: str | None = None) -> str:
        """Use when a user asks whether a specific album exists. Returns a JSON array of {album_title, artist_name} rows filtered by exact album title and optional exact artist."""

        return json.dumps(find_albums(database_path, title, artist))

    @tool
    def lookup_artist(name: str) -> str:
        """Use when a user asks whether an artist exists in the catalog. Returns a JSON array of {artist_name} rows filtered by exact artist name."""

        return json.dumps(find_artists(database_path, name))

    return lookup_track, lookup_album, lookup_artist
