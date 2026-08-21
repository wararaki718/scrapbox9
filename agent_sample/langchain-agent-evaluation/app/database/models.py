from typing import TypedDict


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
