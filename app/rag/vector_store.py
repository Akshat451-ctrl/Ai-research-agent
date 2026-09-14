"""Steps 3 and 4 of RAG: store chunk vectors, then search them.

A vector database keeps each chunk next to its embedding and answers one
question quickly: "which stored vectors point in nearly the same direction
as this one?" Chunks whose meaning is close to the question come back first,
even when they share none of its words.

ChromaDB saves everything under data/vector_db/, so documents are embedded
once and searched many times - no API cost to re-read them on every run.

Search from the command line:
    python -m app.rag.vector_store "What is the biggest risk?"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.llm import LLMError, embed
from app.rag.chunker import Chunk
from app.utils.config import PROJECT_ROOT

VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db"
COLLECTION_NAME = "documents"


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk and how closely it matched the query."""

    text: str
    source: str
    page: int | None
    score: float  # cosine similarity: 1.0 = same meaning, 0 = unrelated

    def citation(self) -> str:
        return f"{self.source}, page {self.page}" if self.page else self.source


@lru_cache(maxsize=1)
def _client() -> chromadb.ClientAPI:
    """Open (or create) the on-disk database once per process."""
    return chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _collection() -> chromadb.Collection:
    # Cosine distance compares the direction of vectors, not their length,
    # which is the right measure for text embeddings.
    return _client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset() -> None:
    """Delete every stored chunk, so a rebuild never keeps stale content."""
    if COLLECTION_NAME in {c.name for c in _client().list_collections()}:
        _client().delete_collection(COLLECTION_NAME)


def add_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Store chunks alongside their vectors and source metadata."""
    _collection().add(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=embeddings,
        # Chroma metadata cannot hold None, so page is only stored for PDFs.
        metadatas=[
            {"source": chunk.source, **({"page": chunk.page} if chunk.page else {})}
            for chunk in chunks
        ],
    )


def count() -> int:
    return _collection().count()


def search(query: str, k: int = 4) -> list[SearchResult]:
    """Return the k chunks whose meaning is closest to the query."""
    total = count()
    if total == 0:
        return []

    query_vector = embed([query], for_query=True)[0]
    result = _collection().query(
        query_embeddings=[query_vector],
        n_results=min(k, total),
        include=["documents", "metadatas", "distances"],
    )

    return [
        SearchResult(
            text=text,
            source=str(metadata["source"]),
            page=metadata.get("page"),
            score=round(1 - distance, 3),
        )
        for text, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]


def main() -> None:
    query = " ".join(sys.argv[1:]) or "What is the biggest operational risk?"
    print(f"Index holds {count()} chunks.\n\nQuery: {query}")

    try:
        results = search(query, k=3)
    except LLMError as error:
        print(f"[failed] {error}")
        return

    if not results:
        print("No results - run `python -m app.rag.ingest` first.")
    for rank, hit in enumerate(results, start=1):
        print(f"\n#{rank}  score={hit.score}  [{hit.citation()}]")
        print(f"    {hit.text[:240]}...")


if __name__ == "__main__":
    main()
