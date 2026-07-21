import requests
from langchain.tools import tool


@tool
def fetch_text(url: str) -> str:
    """Fetch the document from a URL."""
    response = requests.get(url)
    response.raise_for_status()
    return response.text

@tool
def count_lines_containing(text: str, substring: str) -> int:
    """Count lines containing the given substring."""
    return sum(substring in line for line in text.splitlines())


@tool
def find_first_line(text: str, substring: str) -> int | None:
    """Return the 1-based number of the first line containing substring."""
    for line_number, line in enumerate(text.splitlines(), start=1):
        if substring in line:
            return line_number
    return None
