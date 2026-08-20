import json
import sqlite3
import sys
from pathlib import Path

import pytest
import requests
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import (
    DATABASE_URL,
    ensure_database,
    find_albums,
    find_artists,
    find_tracks,
    lookup_purchases,
    refund,
)
from schemas import PurchaseInformation, UserIntent
from tools import create_catalog_tools


def build_catalog_database(tmp_path: Path) -> Path:
    database = tmp_path / "catalog.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE Customer (
            CustomerId INTEGER PRIMARY KEY,
            FirstName TEXT,
            LastName TEXT,
            Phone TEXT
        );
        CREATE TABLE Invoice (
            InvoiceId INTEGER PRIMARY KEY,
            CustomerId INTEGER,
            InvoiceDate TEXT,
            Total REAL
        );
        CREATE TABLE InvoiceLine (
            InvoiceLineId INTEGER PRIMARY KEY,
            InvoiceId INTEGER,
            TrackId INTEGER,
            UnitPrice REAL,
            Quantity INTEGER
        );
        CREATE TABLE Track (
            TrackId INTEGER PRIMARY KEY,
            Name TEXT,
            AlbumId INTEGER
        );
        CREATE TABLE Album (
            AlbumId INTEGER PRIMARY KEY,
            Title TEXT,
            ArtistId INTEGER
        );
        CREATE TABLE Artist (
            ArtistId INTEGER PRIMARY KEY,
            Name TEXT
        );
        INSERT INTO Customer VALUES (1, 'Aaron', 'Mitchell', '+1 204');
        INSERT INTO Invoice VALUES (2, 1, '2009-08-06', 0.99);
        INSERT INTO Artist VALUES (3, 'Led Zeppelin');
        INSERT INTO Artist VALUES (7, 'Pink Floyd');
        INSERT INTO Album VALUES (4, 'IV', 3);
        INSERT INTO Album VALUES (8, 'The Dark Side of the Moon', 7);
        INSERT INTO Track VALUES (5, 'Black Dog', 4);
        INSERT INTO Track VALUES (9, 'Time', 8);
        INSERT INTO InvoiceLine VALUES (6, 2, 5, 0.99, 1);
        """
    )
    connection.commit()
    connection.close()
    return database


def build_catalog_database_with_many_purchases(tmp_path: Path) -> Path:
    database = build_catalog_database(tmp_path)
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        INSERT INTO Artist VALUES (10, 'Extra Artist');
        INSERT INTO Album VALUES (11, 'Extra Album', 10);
        INSERT INTO Track VALUES (12, 'Extra Track', 11);
        """
    )
    for offset in range(25):
        invoice_id = 100 + offset
        invoice_line_id = 100 + offset
        purchase_date = f"2009-08-{7 + offset:02d}"
        connection.execute(
            "INSERT INTO Invoice VALUES (?, 1, ?, 0.99)",
            (invoice_id, purchase_date),
        )
        connection.execute(
            "INSERT INTO InvoiceLine VALUES (?, ?, 12, 0.99, 1)",
            (invoice_line_id, invoice_id),
        )
    connection.commit()
    connection.close()
    return database


def build_refund_database(tmp_path: Path) -> Path:
    database = tmp_path / "refund.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE Invoice (
            InvoiceId INTEGER PRIMARY KEY,
            Total REAL NOT NULL
        );
        CREATE TABLE InvoiceLine (
            InvoiceLineId INTEGER PRIMARY KEY,
            InvoiceId INTEGER NOT NULL,
            UnitPrice REAL NOT NULL,
            Quantity INTEGER NOT NULL
        );
        INSERT INTO Invoice VALUES (237, 1.98);
        INSERT INTO InvoiceLine VALUES (1, 237, 0.99, 2);
        INSERT INTO Invoice VALUES (238, 2.24);
        INSERT INTO InvoiceLine VALUES (2, 238, 1.12, 2);
        """
    )
    connection.close()
    return database


def test_refund_in_mock_mode_reports_total_without_deleting(tmp_path: Path) -> None:
    database = build_refund_database(tmp_path)

    assert refund(database, invoice_id=237, invoice_line_ids=None, mock=True) == 1.98

    connection = sqlite3.connect(database)
    assert connection.execute("SELECT COUNT(*) FROM Invoice WHERE InvoiceId = 237").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM InvoiceLine WHERE InvoiceId = 237").fetchone()[0] == 1
    connection.close()


def test_refund_with_empty_line_ids_is_a_safe_noop(tmp_path: Path) -> None:
    database = build_refund_database(tmp_path)

    assert refund(database, invoice_id=None, invoice_line_ids=[], mock=True) == 0.0


def test_refund_deletes_full_invoice_when_not_mock(tmp_path: Path) -> None:
    database = build_refund_database(tmp_path)

    assert refund(database, invoice_id=237, invoice_line_ids=None, mock=False) == 1.98

    connection = sqlite3.connect(database)
    assert connection.execute("SELECT COUNT(*) FROM Invoice WHERE InvoiceId = 237").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM InvoiceLine WHERE InvoiceId = 237").fetchone()[0] == 0
    connection.close()


def test_refund_deletes_selected_lines_and_returns_line_total(tmp_path: Path) -> None:
    database = build_refund_database(tmp_path)

    assert refund(database, invoice_id=None, invoice_line_ids=[2], mock=False) == 2.24

    connection = sqlite3.connect(database)
    assert connection.execute("SELECT COUNT(*) FROM Invoice WHERE InvoiceId = 238").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM InvoiceLine WHERE InvoiceLineId = 2").fetchone()[0] == 0
    connection.close()


def test_lookup_purchases_uses_customer_identity_and_optional_track(tmp_path: Path) -> None:
    database = build_catalog_database(tmp_path)

    rows = lookup_purchases(
        database,
        "Aaron",
        "Mitchell",
        "+1 204",
        "Black Dog",
        None,
        None,
        None,
    )

    assert rows == [
        {
            "invoice_line_id": 6,
            "track_name": "Black Dog",
            "artist_name": "Led Zeppelin",
            "purchase_date": "2009-08-06",
            "quantity_purchased": 1,
            "price_per_unit": 0.99,
        }
    ]


def test_lookup_purchases_supports_case_insensitive_optional_album_artist_and_date(
    tmp_path: Path,
) -> None:
    database = build_catalog_database(tmp_path)

    rows = lookup_purchases(
        database,
        "aaron",
        "mitchell",
        "+1 204",
        "black dog",
        "iv",
        "led zeppelin",
        "2009-08-06T15:30:00Z",
    )

    assert rows == [
        {
            "invoice_line_id": 6,
            "track_name": "Black Dog",
            "artist_name": "Led Zeppelin",
            "purchase_date": "2009-08-06",
            "quantity_purchased": 1,
            "price_per_unit": 0.99,
        }
    ]


def test_lookup_purchases_caps_results_at_twenty_in_deterministic_order(tmp_path: Path) -> None:
    database = build_catalog_database_with_many_purchases(tmp_path)

    rows = lookup_purchases(
        database,
        "Aaron",
        "Mitchell",
        "+1 204",
        None,
        None,
        None,
        None,
    )

    assert len(rows) == 20
    assert rows[0]["invoice_line_id"] == 6
    assert rows[-1]["invoice_line_id"] == 118


def test_find_tracks_returns_track_album_and_artist(tmp_path: Path) -> None:
    database = build_catalog_database(tmp_path)

    assert find_tracks(database, name="black dog", artist="led zeppelin") == [
        {
            "track_name": "Black Dog",
            "album_title": "IV",
            "artist_name": "Led Zeppelin",
        }
    ]


def test_find_tracks_returns_exact_match_result(tmp_path: Path) -> None:
    database = build_catalog_database(tmp_path)

    assert find_tracks(database, "Black Dog", "Led Zeppelin") == [
        {
            "track_name": "Black Dog",
            "album_title": "IV",
            "artist_name": "Led Zeppelin",
        }
    ]


def test_find_albums_returns_album_and_artist(tmp_path: Path) -> None:
    database = build_catalog_database(tmp_path)

    assert find_albums(database, title="iv", artist="led zeppelin") == [
        {"album_title": "IV", "artist_name": "Led Zeppelin"}
    ]


def test_find_artists_returns_ordered_results_with_optional_filter(tmp_path: Path) -> None:
    database = build_catalog_database(tmp_path)

    assert find_artists(database, name=None) == [
        {"artist_name": "Led Zeppelin"},
        {"artist_name": "Pink Floyd"},
    ]


def test_create_catalog_tools_returns_deterministic_json_results(tmp_path: Path) -> None:
    database = build_catalog_database(tmp_path)

    lookup_track, lookup_album, lookup_artist = create_catalog_tools(database)

    assert lookup_track.name == "lookup_track"
    assert lookup_track.invoke({"name": "Black Dog", "artist": "Led Zeppelin"}) == json.dumps(
        [{"track_name": "Black Dog", "album_title": "IV", "artist_name": "Led Zeppelin"}]
    )
    assert lookup_album.invoke({"title": "IV", "artist": "Led Zeppelin"}) == json.dumps(
        [{"album_title": "IV", "artist_name": "Led Zeppelin"}]
    )
    assert lookup_artist.invoke({"name": "Led Zeppelin"}) == json.dumps(
        [{"artist_name": "Led Zeppelin"}]
    )


def test_purchase_information_tracks_nullable_extraction_fields() -> None:
    purchase_information = PurchaseInformation(
        followup=None,
        invoice_id=None,
        invoice_line_ids=None,
        customer_first_name="Aaron",
        customer_last_name="Mitchell",
        customer_phone="+1 204",
        track_name="Black Dog",
        album_title="IV",
        artist_name="Led Zeppelin",
        purchase_date_iso_8601="2009-08-06",
    )

    assert purchase_information.model_dump() == {
        "followup": None,
        "invoice_id": None,
        "invoice_line_ids": None,
        "customer_first_name": "Aaron",
        "customer_last_name": "Mitchell",
        "customer_phone": "+1 204",
        "track_name": "Black Dog",
        "album_title": "IV",
        "artist_name": "Led Zeppelin",
        "purchase_date_iso_8601": "2009-08-06",
    }


def test_user_intent_accepts_only_supported_routes() -> None:
    assert UserIntent(intent="refund").intent == "refund"
    assert UserIntent(intent="question_answering").intent == "question_answering"

    with pytest.raises(ValidationError):
        UserIntent(intent="other")


class FakeResponse:
    def __init__(
        self,
        *,
        chunks: list[bytes] | None = None,
        error: Exception | None = None,
        status_error: Exception | None = None,
    ) -> None:
        self._chunks = chunks or []
        self._error = error
        self._status_error = status_error
        self.raise_for_status_called = False

    def raise_for_status(self) -> None:
        self.raise_for_status_called = True
        if self._status_error is not None:
            raise self._status_error

    def iter_content(self, chunk_size: int = 8192):
        del chunk_size
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error


def test_ensure_database_returns_existing_file_without_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "chinook.db"
    database.write_bytes(b"existing-data")

    def fail_get(*args, **kwargs):
        raise AssertionError("requests.get should not be called")

    monkeypatch.setattr("database.requests.get", fail_get)

    assert ensure_database(database) == database
    assert database.read_bytes() == b"existing-data"


def test_ensure_database_redownloads_an_empty_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "chinook.db"
    database.touch()
    response = FakeResponse(chunks=[b"chinook"])

    monkeypatch.setattr("database.requests.get", lambda *args, **kwargs: response)

    assert ensure_database(database) == database
    assert database.read_bytes() == b"chinook"
    assert response.raise_for_status_called is True


def test_ensure_database_downloads_atomically_with_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "chinook.db"
    response = FakeResponse(chunks=[b"chi", b"nook"])
    captured: dict[str, object] = {}

    def fake_get(url: str, *, timeout, stream: bool):
        captured["url"] = url
        captured["timeout"] = timeout
        captured["stream"] = stream
        return response

    monkeypatch.setattr("database.requests.get", fake_get)

    assert ensure_database(database) == database
    assert database.read_bytes() == b"chinook"
    assert response.raise_for_status_called is True
    assert captured["url"] == DATABASE_URL
    assert captured["timeout"] == (5, 30)
    assert captured["stream"] is True
    assert sorted(path.name for path in tmp_path.iterdir()) == ["chinook.db"]


def test_ensure_database_cleans_up_partial_file_and_raises_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "chinook.db"
    source_error = requests.RequestException("network dropped")
    response = FakeResponse(chunks=[b"partial"], error=source_error)

    def fake_get(url: str, *, timeout, stream: bool):
        del url, timeout, stream
        return response

    monkeypatch.setattr("database.requests.get", fake_get)

    with pytest.raises(RuntimeError, match="Failed to download Chinook database") as exc_info:
        ensure_database(database)

    assert exc_info.value.__cause__ is source_error
    assert list(tmp_path.iterdir()) == []
