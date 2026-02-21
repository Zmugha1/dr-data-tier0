"""
Local embedding via BGE-Micro-v2 (384 dimensions).
"""

import os
from pathlib import Path
from typing import List, Optional

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


class LocalEmbedder:
    """
    Local embedder using BAAI/bge-micro-v2.
    Produces 384-dimensional vectors.
    """

    MODEL_NAME: str = "TaylorAI/bge-micro-v2"
    DIMENSION: int = 384
    MODEL_SIZE_MB: float = 30.0

    @classmethod
    def is_model_cached(cls) -> bool:
        """Check if BGE-Micro-v2 is already downloaded locally."""
        if SentenceTransformer is None:
            return False
        try:
            _ = SentenceTransformer(cls.MODEL_NAME, local_files_only=True)
            return True
        except Exception:
            return False

    @classmethod
    def get_cache_path(cls) -> str:
        """Return the HuggingFace cache directory path."""
        return os.environ.get(
            "HF_HOME",
            os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub")),
        )

    def __init__(self, cache_dir: Optional[str] = None, local_files_only: bool = False) -> None:
        """Load the BGE-Micro-v2 model."""
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        kwargs = {}
        if cache_dir:
            kwargs["cache_folder"] = cache_dir
        if local_files_only:
            kwargs["local_files_only"] = True
        try:
            self._model = SentenceTransformer(self.MODEL_NAME, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model {self.MODEL_NAME}: {e}") from e

    def embed(self, text: str) -> List[float]:
        """
        Embed a single text into a 384-dim vector.

        Args:
            text: Input text to embed.

        Returns:
            List of 384 floats.
        """
        if not text or not text.strip():
            return [0.0] * self.DIMENSION
        try:
            vector = self._model.encode(text, convert_to_numpy=True)
            return vector.tolist()
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}") from e

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts efficiently.

        Args:
            texts: List of input texts.

        Returns:
            List of 384-dim vectors.
        """
        if not texts:
            return []
        cleaned = [t.strip() if t else "" for t in texts]
        try:
            vectors = self._model.encode(cleaned, convert_to_numpy=True)
            return [v.tolist() for v in vectors]
        except Exception as e:
            raise RuntimeError(f"Batch embedding failed: {e}") from e
