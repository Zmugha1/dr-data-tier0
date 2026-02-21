"""
Ollama vision client for image-to-text description.
"""

import base64
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore

from core.config import OLLAMA_HOST, VISION_MODEL


class OllamaVisionClient:
    """
    Client for Ollama vision models to generate text descriptions from images.
    """

    MODEL: str = VISION_MODEL

    def __init__(self, host: Optional[str] = None, model: Optional[str] = None) -> None:
        """Initialize vision client."""
        if requests is None:
            raise ImportError("requests is required. Install with: pip install requests")
        self._host = (host or OLLAMA_HOST).rstrip("/")
        self._model = model or self.MODEL

    def describe_image(self, image_path: Path) -> str:
        """
        Generate text description from image using Ollama vision model.

        Args:
            image_path: Path to image file (PNG, JPG, JPEG).

        Returns:
            Text description of the image content.
        """
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        response = requests.post(
            f"{self._host}/api/generate",
            json={
                "model": self._model,
                "prompt": "Describe this image in detail. Extract any text, tables, charts, or data visible.",
                "images": [image_b64],
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
