"""Build the search index from everything in data/documents/.

    python -m app.rag.ingest

This is the OFFLINE half of RAG - run it whenever documents change. It reads,
chunks, embeds and stores. The ONLINE half (vector_store.search) runs on every
question and only embeds the question itself, which is why search is cheap.
"""

from __future__ import annotations

from app.llm import LLMError, embed
from app.rag import vector_store
from app.rag.chunker import chunk_documents
from app.rag.loader import DOCUMENTS_DIR, load_documents


def ingest() -> int:
    """Rebuild the index from scratch. Returns the number of chunks stored."""
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        print(f"No .pdf, .txt or .md files found in {DOCUMENTS_DIR}")
        return 0

    files = sorted({document.source for document in documents})
    print(f"[load]   {len(documents)} section(s) from {len(files)} file(s): {', '.join(files)}")

    chunks = chunk_documents(documents)
    sizes = [len(chunk.text) for chunk in chunks]
    print(f"[chunk]  {len(chunks)} chunks, {min(sizes)}-{max(sizes)} characters each")

    vectors = embed([chunk.text for chunk in chunks])
    print(f"[embed]  {len(vectors)} vectors x {len(vectors[0])} dimensions")

    vector_store.reset()
    vector_store.add_chunks(chunks, vectors)
    print(f"[store]  {vector_store.count()} chunks saved to {vector_store.VECTOR_DB_DIR}")

    return len(chunks)


def main() -> None:
    try:
        ingest()
    except LLMError as error:
        print(f"[failed] {error}")


if __name__ == "__main__":
    main()
