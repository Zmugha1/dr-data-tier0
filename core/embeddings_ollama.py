"""
Ollama-based embeddings fallback when BGE-Micro is unavailable.
Uses nomic-embed-text (768 dimensions). Run: ollama pull nomic-embed-text
"""

from typing import List, Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore

from core.config import OLLAMA_HOST


class OllamaEmbedder:
    """
    Ollama-based embedder fallback.
    Produces 768-dimensional vectors (nomic-embed-text).
    """

    MODEL_NAME: str = "nomic-embed-text"
    DIMENSION: int = 768

    def __init__(self, host: Optional[str] = None) -> None:
        """Initialize and verify Ollama is running with embed model."""
        if requests is None:
            raise ImportError("requests is required. Install with: pip install requests")
        self._host = (host or OLLAMA_HOST).rstrip("/")
        try:
            resp = requests.get(f"{self._host}/api/tags", timeout=5)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama not responding at {self._host}")
            models = resp.json().get("models", [])
            if not any(self.MODEL_NAME in m.get("name", "") for m in models):
                raise RuntimeError(
                    f"Model {self.MODEL_NAME} not found. Run: ollama pull {self.MODEL_NAME}"
                )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Cannot connect to Ollama at {self._host}: {e}") from e

    def embed(self, text: str) -> List[float]:
        """Embed a single text into a 768-dim vector."""
        if not text or not text.strip():
            return [0.0] * self.DIMENSION
        try:
            resp = requests.post(
                f"{self._host}/api/embeddings",
                json={"model": self.MODEL_NAME, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as e:
            raise RuntimeError(f"Ollama embedding failed: {e}") from e

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts (sequential calls to Ollama)."""
        if not texts:
            return []
        return [self.embed(t or "") for t in texts]
