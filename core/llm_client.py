"""
Ollama API client for local Phi-3 inference.
"""

from typing import List, Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore

from core.config import MODEL_NAME, OLLAMA_HOST


class OllamaClient:
    """
    Client for Ollama HTTP API. Uses Phi-3 mini with low temperature.
    """

    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 512

    def __init__(self, host: Optional[str] = None, model: Optional[str] = None) -> None:
        """
        Initialize Ollama client.

        Args:
            host: Ollama API base URL. Defaults to OLLAMA_HOST.
            model: Model name. Defaults to MODEL_NAME (phi3:mini).
        """
        if requests is None:
            raise ImportError("requests is required. Install with: pip install requests")
        self._host = (host or OLLAMA_HOST).rstrip("/")
        self._model = model or MODEL_NAME

    def is_available(self) -> bool:
        """
        Check if Ollama is running and the model is available.

        Returns:
            True if API responds and model exists.
        """
        try:
            r = requests.get(f"{self._host}/api/tags", timeout=5)
            if r.status_code != 200:
                return False
            data = r.json()
            models = data.get("models", [])
            return any(m.get("name", "").startswith(self._model) for m in models)
        except Exception:
            return False

    def chat(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Send a chat prompt to Ollama.

        Args:
            prompt: User message or full prompt.
            context: Optional RAG context to prepend.

        Returns:
            Model response text.
        """
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"

        try:
            payload = {
                "model": self._model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.TEMPERATURE,
                    "num_predict": self.MAX_TOKENS,
                },
            }
            r = requests.post(
                f"{self._host}/api/generate",
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}") from e
        except (KeyError, TypeError) as e:
            raise RuntimeError(f"Unexpected Ollama response: {e}") from e
