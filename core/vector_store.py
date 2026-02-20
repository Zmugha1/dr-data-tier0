"""
ChromaDB wrapper for local vector storage.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import chromadb
except ImportError:
    chromadb = None  # type: ignore

from core.config import CHROMA_PATH


class LocalVectorStore:
    """
    Persistent ChromaDB vector store for document retrieval.
    """

    COLLECTION_NAME: str = "dr_data_docs"

    def __init__(self, persist_path: Optional[Path] = None) -> None:
        """
        Initialize ChromaDB persistent client.

        Args:
            persist_path: Directory for ChromaDB storage. Defaults to CHROMA_PATH.
        """
        if chromadb is None:
            raise ImportError("chromadb is required. Install with: pip install chromadb")
        path = Path(persist_path) if persist_path else Path(CHROMA_PATH)
        try:
            self._client = chromadb.PersistentClient(path=str(path))
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
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
