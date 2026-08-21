import sqlite3
from pathlib import Path


def fetch_rows(database: Path, query: str, parameters: list[str]) -> list[sqlite3.Row]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(query, tuple(parameters)).fetchall()
    finally:
        connection.close()
