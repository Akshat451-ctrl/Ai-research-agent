"""Step 2 of RAG: split documents into chunks.

Why not embed a whole document? An embedding squeezes any amount of text into
a fixed 768 numbers. Squeeze a 40-page report into 768 numbers and the one
paragraph you need gets averaged away. Small chunks keep each vector focused
on one idea, so a search can land on exactly the passage that matters.

Strategy, in order of preference:
  1. Never mix sections - a heading always starts a new chunk.
  2. Pack whole paragraphs together until the chunk is full.
  3. Only a paragraph too long to fit (common in PDFs, where a page often
     arrives as one block) gets cut with a sliding window. Those cuts can land
     mid-sentence, so neighbouring windows OVERLAP to keep every sentence
     whole in at least one of them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.loader import Document

CHUNK_SIZE = 800      # characters - roughly 150 words
CHUNK_OVERLAP = 150   # characters repeated between sliding windows

# One or more blank lines separate paragraphs.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    """A searchable slice of a document."""

    id: str              # stable, so re-indexing the same file gives the same ids
    text: str
    source: str
    page: int | None


def _paragraphs(text: str) -> list[str]:
    """Split on blank lines, then tidy whitespace inside each paragraph."""
    return [" ".join(block.split()) for block in _PARAGRAPH_BREAK.split(text) if block.strip()]


def _sliding_window(text: str, size: int, overlap: int) -> list[str]:
    """Cut one long paragraph into overlapping windows, breaking only at spaces."""
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))

        # Don't cut a word in half: step back to the last space in the window.
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        pieces.append(text[start:end].strip())
        if end >= len(text):
            break

        # Begin the next window `overlap` characters before this one ended,
        # moved forward to the next word boundary.
        start = max(end - overlap, start + 1)
        next_space = text.find(" ", start, end)
        if next_space != -1:
            start = next_space + 1

    return [piece for piece in pieces if piece]


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Group paragraphs into chunks of at most `size` characters."""
    pieces: list[str] = []
    current = ""

    for paragraph in _paragraphs(text):
        # A Markdown heading marks a new topic - close the chunk in progress.
        if paragraph.startswith("#") and current:
            pieces.append(current)
            current = ""

        combined = f"{current} {paragraph}".strip()

        if len(combined) <= size:
            # Still fits: keep packing.
            current = combined
        elif len(paragraph) <= size:
            # This paragraph fits on its own, just not with what came before.
            pieces.append(current)
            current = paragraph
        else:
            # Too long for any chunk. Window over it, keeping whatever was in
            # progress (often its heading) attached to the first window.
            pieces.extend(_sliding_window(combined, size, overlap))
            current = ""

    if current:
        pieces.append(current)
    return pieces


def chunk_documents(
    documents: list[Document],
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split every document into chunks that remember their source."""
    chunks: list[Chunk] = []
    for document in documents:
        for index, piece in enumerate(_split_text(document.text, size, overlap)):
            chunks.append(
                Chunk(
                    id=f"{document.source}::p{document.page or 0}::c{index}",
                    text=piece,
                    source=document.source,
                    page=document.page,
                )
            )
    return chunks
