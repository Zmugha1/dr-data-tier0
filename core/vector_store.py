"""
ChromaDB wrapper for local vector storage.
Simple VectorStore with explicit embedding function.
"""

import gc
import json
import shutil
from pathlib import Path
from typing import Any, Dict

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


def safe_vector_store_init(collection_name: str = "dr_data_default") -> "VectorStore":
    """
    Check for corruption BEFORE instantiating client, delete if needed, then create fresh.
    """
    db_path = Path("data/vector_db")
    db_path.mkdir(parents=True, exist_ok=True)

    if (db_path / "chroma.sqlite3").exists():
        try:
            test_client = chromadb.PersistentClient(path=str(db_path))
            test_client.list_collections()
            del test_client
        except Exception:
            shutil.rmtree(db_path, ignore_errors=True)
            gc.collect()
            db_path.mkdir(parents=True, exist_ok=True)

    return VectorStore(collection_name=collection_name)


class VectorStore:
    """Simple VectorStore with explicit embedding function."""

    def __init__(self, collection_name: str = "dr_data_default"):
        self.persist_dir = Path("data/vector_db")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Explicit embedding function
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_func,
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_func,
                metadata={"hnsw:space": "cosine"},
            )

    def add_document(self, doc_data: Dict[str, Any]) -> int:
        """Add document chunks to vector store with idempotency."""
        chunks = doc_data["chunks"]
        if not chunks:
            return 0

        # Idempotency: skip chunks already in DB
        existing_ids = set()
        try:
            existing = self.collection.get()
            if existing and "ids" in existing:
                existing_ids = set(existing["ids"])
        except Exception:
            pass

        new_chunks = [
            c
            for c in chunks
            if c.get("content_hash") and c["content_hash"] not in existing_ids
        ]
        if not new_chunks:
            return 0

        ids = [c["content_hash"] for c in new_chunks]
        texts = [c["text"] for c in new_chunks]
        metadatas = [
            {
                "doc_hash": c["doc_hash"],
                "chunk_index": c["chunk_index"],
                "source": doc_data["filename"],
                "char_start": c.get("char_start", 0),
                "char_end": c.get("char_end", 0),
            }
            for c in new_chunks
        ]

        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
        return len(new_chunks)

    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the vector store."""
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get collection stats."""
        return {
            "chunks": self.collection.count(),
            "status": "Active",
            "path": str(self.persist_dir),
        }

    def persist(self) -> Dict[str, Any] | None:
        """Write manifest for audit."""
        count = self.collection.count()
        manifest = {
            "collection": self.collection_name,
            "embedding_model": "all-MiniLM-L6-v2",
            "chunks_indexed": count,
            "storage_path": str(self.persist_dir),
        }
        with open(self.persist_dir / "vector_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest


def get_vector_store() -> VectorStore:
    """Get initialized VectorStore for RAG pages."""
    return safe_vector_store_init()
