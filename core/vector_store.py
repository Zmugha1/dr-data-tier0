"""
ChromaDB wrapper for local vector storage.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None  # type: ignore
    Settings = None  # type: ignore
    embedding_functions = None  # type: ignore

from core.config import CHROMA_PATH


class LocalVectorStore:
    """
    Persistent ChromaDB vector store for document retrieval.
    """

    COLLECTION_NAME: str = "dr_data_docs"

    def __init__(
        self,
        persist_path: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Initialize ChromaDB persistent client.

        Args:
            persist_path: Directory for ChromaDB storage. Defaults to CHROMA_PATH.
            collection_name: Override collection (e.g. for different embedding dimensions).
        """
        if chromadb is None:
            raise ImportError("chromadb is required. Install with: pip install chromadb")
        path = Path(persist_path) if persist_path else Path(CHROMA_PATH)
        path.mkdir(parents=True, exist_ok=True)
        coll_name = collection_name or self.COLLECTION_NAME
        try:
            self._client = chromadb.PersistentClient(path=str(path))
            self._collection = self._client.get_or_create_collection(
                name=coll_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}") from e

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """
        Add a document to the vector store.

        Args:
            doc_id: Unique identifier for the document.
            text: Document content (stored in metadata for retrieval).
            metadata: Optional metadata dict. 'text' will be set from text.
            embedding: Optional precomputed embedding. If None, caller must provide.
        """
        meta = dict(metadata or {})
        meta["text"] = text

        try:
            if embedding is not None:
                self._collection.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    metadatas=[meta],
                )
            else:
                self._collection.upsert(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[meta],
                )
        except Exception as e:
            raise RuntimeError(f"Failed to add document {doc_id}: {e}") from e

    def query(
        self,
        query_embedding: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        n_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents.

        Args:
            query_embedding: Precomputed query vector.
            query_text: Query text (requires collection to support document query).
            n_results: Number of results to return.

        Returns:
            List of dicts with 'id', 'metadata', 'distance'.
        """
        try:
            if query_embedding is not None:
                results = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["metadatas", "documents", "distances"],
                )
            elif query_text is not None:
                results = self._collection.query(
                    query_texts=[query_text],
                    n_results=n_results,
                    include=["metadatas", "documents", "distances"],
                )
            else:
                raise ValueError("Either query_embedding or query_text must be provided")
        except Exception as e:
            raise RuntimeError(f"Query failed: {e}") from e

        ids = results["ids"][0] if results["ids"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        documents = results.get("documents", [[]])
        docs_list = documents[0] if documents else []
        distances = results.get("distances", [[]])
        dist_list = distances[0] if distances else []

        return [
            {
                "id": ids[i] if i < len(ids) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "document": docs_list[i] if i < len(docs_list) else "",
                "distance": dist_list[i] if i < len(dist_list) else None,
            }
            for i in range(min(len(ids), n_results))
        ]


class VectorStore:
    """Steps 4-5: Deterministic embedding and ChromaDB storage for RAG pipeline."""

    def __init__(self, collection_name: str = "dr_data_default", reset_if_conflict: bool = False):
        if chromadb is None or embedding_functions is None:
            raise ImportError("chromadb required. Install with: pip install chromadb sentence-transformers")
        self.persist_dir = Path("data/vector_db")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model = "all-MiniLM-L6-v2"

        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )

        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False) if Settings else None,
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
        except Exception as e:
            err_lower = str(e).lower()
            if "different settings" in err_lower or "already exists" in err_lower or "conflict" in err_lower:
                if reset_if_conflict:
                    shutil.rmtree(self.persist_dir, ignore_errors=True)
                    self.persist_dir.mkdir(parents=True, exist_ok=True)
                    self.client = chromadb.PersistentClient(
                        path=str(self.persist_dir),
                        settings=Settings(anonymized_telemetry=False) if Settings else None,
                    )
                    self.collection = self.client.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_func,
                        metadata={"hnsw:space": "cosine"},
                    )
                else:
                    raise RuntimeError(
                        f"Vector DB settings conflict. Delete '{self.persist_dir}' manually, "
                        f"or use VectorStore(reset_if_conflict=True). Original: {e}"
                    ) from e
            else:
                raise

    def add_document(self, doc_data: Dict[str, Any]) -> None:
        """Step 4: Generate embeddings with idempotency."""
        chunks = doc_data["chunks"]

        existing_ids: set = set()
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
            return

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

    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Retrieve context for RAG."""
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    def persist(self) -> Dict[str, Any]:
        """Step 5: Validation and manifest export."""
        count = self.collection.count()
        manifest = {
            "collection": self.collection_name,
            "embedding_model": self.embedding_model,
            "chunks_indexed": count,
            "storage_path": str(self.persist_dir),
        }
        with open(self.persist_dir / "vector_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest
