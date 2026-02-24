"""Simple RAG implementation with error handling"""
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st


class SimpleRAG:
    """File-based RAG with JSON storage (no database locks)"""

    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.index_file = self.data_dir / "document_index.json"
        self.documents = self._load_index()

    def _load_index(self) -> Dict:
        """Load document index"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_index(self):
        """Save document index"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate similarity"""
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def add_document(self, name: str, chunks: List[Dict]) -> str:
        """Add document with pre-computed embeddings"""
        doc_id = hashlib.md5(name.encode()).hexdigest()[:12]
        self.documents[doc_id] = {
            "name": name,
            "chunks": chunks
        }
        self._save_index()
        return doc_id

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search documents"""
        results = []
        for doc_id, doc in self.documents.items():
            for chunk in doc["chunks"]:
                score = self.cosine_similarity(query_embedding, chunk["embedding"])
                results.append({
                    "doc_name": doc["name"],
                    "text": chunk["text"],
                    "score": score
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_stats(self) -> Dict:
        """Get document stats"""
        total_chunks = sum(len(d["chunks"]) for d in self.documents.values())
        return {
            "documents": len(self.documents),
            "chunks": total_chunks
        }

    def clear_all(self):
        """Delete all documents"""
        self.documents = {}
        self._save_index()

    def list_documents(self) -> List[Dict]:
        """List all documents"""
        return [
            {"id": k, "name": v["name"], "chunks": len(v["chunks"])}
            for k, v in self.documents.items()
        ]
