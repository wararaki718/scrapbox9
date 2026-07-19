import requests
from langchain.tools import tool


@tool
def fetch_text(url: str) -> str:
    """Fetch the document from a URL."""
    response = requests.get(url)
    response.raise_for_status()
    return response.text
