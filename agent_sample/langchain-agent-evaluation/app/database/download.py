from pathlib import Path

import requests

DATABASE_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
DOWNLOAD_TIMEOUT = (5, 30)


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
