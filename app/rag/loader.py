"""Step 1 of RAG: load raw text out of files.

Each PDF page becomes its own Document so that, later, every answer can cite
the exact page it came from. Text and Markdown files have no pages, so they
become a single Document each.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.utils.config import PROJECT_ROOT

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class Document:
    """A piece of source text plus where it came from."""

    text: str
    source: str          # file name, e.g. "annual_report.pdf"
    page: int | None     # 1-based page number for PDFs, None otherwise


def _load_pdf(path: Path) -> list[Document]:
    """One Document per non-empty page."""
    documents: list[Document] = []
    for number, page in enumerate(PdfReader(path).pages, start=1):
        text = (page.extract_text() or "").strip()
        # Scanned PDFs are images with no text layer - they come back empty.
        if text:
            documents.append(Document(text=text, source=path.name, page=number))
    return documents


def _load_text(path: Path) -> list[Document]:
    """A whole .txt or .md file as a single Document."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [Document(text=text, source=path.name, page=None)] if text else []


def load_documents(folder: Path = DOCUMENTS_DIR) -> list[Document]:
    """Load every supported file in `folder` (not recursive)."""
    documents: list[Document] = []
    for path in sorted(folder.iterdir()):
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        if suffix == ".pdf":
            documents.extend(_load_pdf(path))
        else:
            documents.extend(_load_text(path))
    return documents
