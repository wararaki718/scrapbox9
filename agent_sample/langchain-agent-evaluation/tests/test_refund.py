import sqlite3
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import app.refund as refund_module
from app.schemas import PurchaseInformation


class FakeStructuredRunnable:
    def __init__(self, result: PurchaseInformation) -> None:
        self.result = result
        self.calls: list[list[object]] = []

    def invoke(self, messages: list[object]) -> PurchaseInformation:
        self.calls.append(messages)
        return self.result


class FakeModel:
    def __init__(self, result: PurchaseInformation) -> None:
        self.result = result
        self.schema = None
        self.runnable = FakeStructuredRunnable(result)

    def with_structured_output(self, schema):
        self.schema = schema
        return self.runnable


def build_catalog_database(tmp_path: Path) -> Path:
    database = tmp_path / 'catalog.db'
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
        INSERT INTO Album VALUES (4, 'IV', 3);
        INSERT INTO Track VALUES (5, 'Black Dog', 4);
        INSERT INTO InvoiceLine VALUES (6, 2, 5, 0.99, 1);
        """
    )
    connection.close()
    return database


def test_next_refund_step_prefers_refund_for_invoice_or_selected_lines() -> None:
    assert refund_module.next_refund_step(237, None, None, None, None) == 'refund'
    assert refund_module.next_refund_step(None, [6], None, None, None) == 'refund'


def test_next_refund_step_uses_lookup_for_complete_identity() -> None:
    assert refund_module.next_refund_step(None, [], 'Aaron', 'Mitchell', '+1 204') == 'lookup'


def test_next_refund_step_uses_respond_for_incomplete_identity() -> None:
    assert refund_module.next_refund_step(None, [], 'Aaron', None, '+1 204') == 'respond'
    assert refund_module.next_refund_step(None, [], ' ', 'Mitchell', '+1 204') == 'respond'


def test_create_refund_graph_looks_up_purchases_and_records_invoice_line_ids(tmp_path: Path) -> None:
    database = build_catalog_database(tmp_path)
    model = FakeModel(
        PurchaseInformation(
            customer_first_name='Aaron',
            customer_last_name='Mitchell',
            customer_phone='+1 204',
            track_name='Black Dog',
        )
    )

    graph = refund_module.create_refund_graph(model, database)

    result = graph.invoke({'messages': [HumanMessage(content='Please help with a refund.')]})

    assert model.schema is PurchaseInformation
    assert result['first_name'] == 'Aaron'
    assert result['last_name'] == 'Mitchell'
    assert result['phone'] == '+1 204'
    assert result['invoice_line_ids'] == [6]
    assert 'I found these matching purchases:' in result['followup']
    assert 'Black Dog' in result['followup']
    assert 'invoice_line_id' in result['followup']


def test_create_refund_graph_returns_fixed_no_purchase_message(tmp_path: Path) -> None:
    database = build_catalog_database(tmp_path)
    model = FakeModel(
        PurchaseInformation(
            customer_first_name='Wrong',
            customer_last_name='Customer',
            customer_phone='000',
        )
    )

    graph = refund_module.create_refund_graph(model, database)

    result = graph.invoke({'messages': [HumanMessage(content='Find my purchase.')]})

    assert result['invoice_line_ids'] == []
    assert result['followup'] == (
        "I couldn't find any matching purchases. Please verify the customer's first name, "
        'last name, phone number, and any track, album, artist, or purchase date details.'
    )


def test_create_refund_graph_previews_refund_outside_prod(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_refund(database: Path, invoice_id: int | None, invoice_line_ids: list[int] | None, *, mock: bool) -> float:
        captured['database'] = database
        captured['invoice_id'] = invoice_id
        captured['invoice_line_ids'] = invoice_line_ids
        captured['mock'] = mock
        return 1.98

    monkeypatch.setattr(refund_module, 'apply_refund', fake_refund)
    model = FakeModel(PurchaseInformation(invoice_id=237))
    graph = refund_module.create_refund_graph(model, tmp_path / 'refund.db')

    result = graph.invoke({'messages': [HumanMessage(content='Refund invoice 237.')]})

    assert captured == {
        'database': tmp_path / 'refund.db',
        'invoice_id': 237,
        'invoice_line_ids': None,
        'mock': True,
    }
    assert result['followup'] == (
        'Previewed a refund total of $1.98. No database changes were made because env is not prod.'
    )


def test_create_refund_graph_executes_refund_in_prod(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_refund(database: Path, invoice_id: int | None, invoice_line_ids: list[int] | None, *, mock: bool) -> float:
        captured['database'] = database
        captured['invoice_id'] = invoice_id
        captured['invoice_line_ids'] = invoice_line_ids
        captured['mock'] = mock
        return 2.24

    monkeypatch.setattr(refund_module, 'apply_refund', fake_refund)
    model = FakeModel(PurchaseInformation(invoice_line_ids=[2]))
    graph = refund_module.create_refund_graph(model, tmp_path / 'refund.db')

    result = graph.invoke(
        {'messages': [HumanMessage(content='Refund line 2.')]},
        config={'configurable': {'env': 'prod'}},
    )

    assert captured == {
        'database': tmp_path / 'refund.db',
        'invoice_id': None,
        'invoice_line_ids': [2],
        'mock': False,
    }
    assert result['followup'] == 'Refunded $2.24 successfully.'


def test_create_refund_graph_uses_model_followup_when_more_identity_is_needed(tmp_path: Path) -> None:
    model = FakeModel(
        PurchaseInformation(
            customer_first_name='Aaron',
            followup='Please share the customer last name and phone number to continue.',
        )
    )
    graph = refund_module.create_refund_graph(model, tmp_path / 'catalog.db')

    result = graph.invoke({'messages': [HumanMessage(content='I need a refund.')]})

    assert result['followup'] == 'Please share the customer last name and phone number to continue.'


def test_create_refund_graph_uses_fixed_followup_when_model_has_none(tmp_path: Path) -> None:
    model = FakeModel(PurchaseInformation(customer_first_name='Aaron'))
    graph = refund_module.create_refund_graph(model, tmp_path / 'catalog.db')

    result = graph.invoke({'messages': [HumanMessage(content='I need a refund.')]})

    assert result['followup'] == (
        "To process the refund, please share the customer's first name, last name, and phone number "
        "so I can look up the purchase."
    )


def test_create_refund_graph_clears_stale_followup_when_current_extraction_has_none(
    tmp_path: Path,
) -> None:
    model = FakeModel(PurchaseInformation(customer_last_name='Mitchell'))
    graph = refund_module.create_refund_graph(model, tmp_path / 'catalog.db')

    result = graph.invoke(
        {
            'messages': [HumanMessage(content='My last name is Mitchell.')],
            'first_name': 'Aaron',
            'followup': 'Please share the customer last name and phone number to continue.',
        }
    )

    assert result['followup'] == (
        "To process the refund, please share the customer's first name, last name, and phone number "
        "so I can look up the purchase."
    )


def test_create_refund_graph_clears_stale_line_ids_when_invoice_id_is_selected(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_refund(database: Path, invoice_id: int | None, invoice_line_ids: list[int] | None, *, mock: bool) -> float:
        captured['database'] = database
        captured['invoice_id'] = invoice_id
        captured['invoice_line_ids'] = invoice_line_ids
        captured['mock'] = mock
        return 1.98

    monkeypatch.setattr(refund_module, 'apply_refund', fake_refund)
    model = FakeModel(PurchaseInformation(invoice_id=237))
    graph = refund_module.create_refund_graph(model, tmp_path / 'refund.db')

    graph.invoke(
        {
            'messages': [HumanMessage(content='Refund invoice 237.')],
            'invoice_line_ids': [6],
        }
    )

    assert captured == {
        'database': tmp_path / 'refund.db',
        'invoice_id': 237,
        'invoice_line_ids': None,
        'mock': True,
    }


def test_create_refund_graph_does_not_auto_refund_from_lookup_results_on_later_turn(
    tmp_path: Path, monkeypatch
) -> None:
    database = build_catalog_database(tmp_path)

    def fail_refund(*args, **kwargs):
        raise AssertionError('refund should not be called without a newly selected refund target')

    monkeypatch.setattr(refund_module, 'apply_refund', fail_refund)
    model = FakeModel(PurchaseInformation())
    graph = refund_module.create_refund_graph(model, database)

    result = graph.invoke(
        {
            'messages': [HumanMessage(content='What did I buy?')],
            'invoice_line_ids': [6],
            'first_name': 'Aaron',
            'last_name': 'Mitchell',
            'phone': '+1 204',
            'track_name': 'Black Dog',
        }
    )

    assert result['invoice_line_ids'] == [6]
    assert 'I found these matching purchases:' in result['followup']
