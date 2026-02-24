"""LLM Management with automatic model availability checking"""
import ollama
import requests
from typing import List, Dict
from setup_checker import get_setup_checker


class LLMManager:
    """Manages LLM interactions with safety checks"""

    def __init__(self):
        self.checker = get_setup_checker()
        self.host = "http://localhost:11434"

    def is_ollama_running(self) -> bool:
        """Check if Ollama server is running"""
        self.checker.check_ollama()
        return self.checker.status["ollama_running"]

    def ensure_model(self, model_name: str) -> bool:
        """Ensure model is available before use. Returns False if Ollama down or model missing."""
        if not self.is_ollama_running():
            return False
        self.checker.check_models()
        return self.checker.status["models"].get(model_name, False)

    def embed(self, text: str, model: str = "nomic-embed-text:latest") -> List[float]:
        """Get embeddings with safety check"""
        if not self.is_ollama_running():
            raise RuntimeError("Ollama not running. Start it with: ollama serve")
        if not self.ensure_model(model):
            raise RuntimeError(f"Model {model} not installed. Run setup first.")

        response = ollama.embeddings(model=model, prompt=text)
        return response['embedding']

    def generate(self, prompt: str, model: str = "llama3.1:8b",
                 temperature: float = 0.1) -> str:
        """Generate text with safety check"""
        if not self.is_ollama_running():
            raise RuntimeError("Ollama not running. Start it with: ollama serve")
        if not self.ensure_model(model):
            raise RuntimeError(f"Model {model} not installed. Run setup first.")

        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature}
        )
        return response['response']

    def list_models(self) -> List[Dict]:
        """List all managed models and their status"""
        self.checker.check_models()
        result = []
        for model_id, info in self.checker.REQUIRED_MODELS.items():
            result.append({
                "id": model_id,
                "name": info["name"],
                "available": self.checker.status["models"].get(model_id, False)
            })
        return result
