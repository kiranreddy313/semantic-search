from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_document(path):
    """Return a list of (page_number, text) pairs. Text files are one 'page' with no number."""
    path = Path(path)

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append((page_number, text))
        return pages

    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(None, text)]