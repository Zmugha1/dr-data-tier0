"""Setup verification - ensures everything is installed before app runs"""
import subprocess
import requests
import sys
from typing import Dict, List, Tuple
import streamlit as st


class SetupChecker:
    """Checks all dependencies before app startup"""

    REQUIRED_MODELS = {
        "nomic-embed-text:latest": {
            "name": "Nomic Embed Text",
            "purpose": "Document embeddings (CRITICAL)",
            "size": "~275MB",
            "auto_install": True
        },
        "llama3.1:8b": {
            "name": "Llama 3.1 8B",
            "purpose": "General chat & reasoning",
            "size": "~4.7GB",
            "auto_install": True
        },
        "phi4:latest": {
            "name": "Phi-4",
            "purpose": "Fast extraction & analysis",
            "size": "~2.7GB",
            "auto_install": True
        },
        "qwen2.5:7b": {
            "name": "Qwen 2.5 7B",
            "purpose": "Reasoning (laptop-friendly)",
            "size": "~4.5GB",
            "auto_install": False
        },
        "qwen2.5:14b": {
            "name": "Qwen 2.5 14B",
            "purpose": "Complex reasoning (needs 10GB+ VRAM)",
            "size": "~9GB",
            "auto_install": False
        }
    }

    # (pip_package_name, import_module_name) - some differ (e.g. pillow -> PIL)
    PYTHON_PACKAGES = [
        ("pdfplumber", "pdfplumber"),
        ("pytesseract", "pytesseract"),
        ("Pillow", "PIL"),  # pip package is Pillow (capital P)
        ("sentence-transformers", "sentence_transformers"),
        ("plotly", "plotly"),
        ("networkx", "networkx")
    ]

    PACKAGE_DESCRIPTIONS = {
        "pdfplumber": "PDF text extraction",
        "pytesseract": "OCR for scanned PDFs",
        "Pillow": "Image processing",
        "sentence-transformers": "Fallback embeddings",
        "plotly": "Visualizations",
        "networkx": "Graph analysis"
    }

    def __init__(self):
        self.ollama_host = "http://localhost:11434"
        self.status = {
            "ollama_running": False,
            "models": {},
            "packages": {},
            "ready": False
        }

    def check_ollama(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            self.status["ollama_running"] = response.status_code == 200
            return self.status["ollama_running"]
        except Exception:
            self.status["ollama_running"] = False
            return False

    def check_models(self) -> Dict[str, bool]:
        """Check which models are installed"""
        if not self.status["ollama_running"]:
            return {k: False for k in self.REQUIRED_MODELS}

        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            installed = [m['name'] for m in response.json().get('models', [])]

            for model_id in self.REQUIRED_MODELS:
                # Check if model exists (handle tag variations)
                found = any(
                    model_id in m or m.replace(':latest', '') in model_id
                    for m in installed
                )
                self.status["models"][model_id] = found

            return self.status["models"]
        except Exception as e:
            st.error(f"Error checking models: {e}")
            return {k: False for k in self.REQUIRED_MODELS}

    def check_python_packages(self) -> Dict[str, bool]:
        """Check optional Python packages"""
        for pip_name, import_name in self.PYTHON_PACKAGES:
            try:
                __import__(import_name)
                self.status["packages"][pip_name] = True
            except ImportError:
                self.status["packages"][pip_name] = False
        return self.status["packages"]

    def install_model(self, model_id: str) -> Tuple[bool, str]:
        """Install a model via Ollama"""
        try:
            result = subprocess.run(
                ["ollama", "pull", model_id],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes for large models
            )
            if result.returncode == 0:
                return True, f"Successfully installed {model_id}"
            else:
                return False, result.stderr or "Unknown error"
        except subprocess.TimeoutExpired:
            return False, "Installation timed out (model too large or network slow)"
        except Exception as e:
            return False, str(e)

    def is_ready(self) -> bool:
        """Check if minimum requirements are met"""
        if not self.status["ollama_running"]:
            return False

        # Must have embeddings and at least one chat model
        required = ["nomic-embed-text:latest", "llama3.1:8b"]  # phi4 also accepted by app
        ready = all(self.status["models"].get(m, False) for m in required)

        self.status["ready"] = ready
        return ready

    def get_setup_commands(self) -> str:
        """Generate terminal commands for manual setup"""
        commands = []

        if not self.status["ollama_running"]:
            commands.append("# Start Ollama server:")
            commands.append("ollama serve")
            commands.append("")

        # Missing models
        for model_id, info in self.REQUIRED_MODELS.items():
            if not self.status["models"].get(model_id, False):
                commands.append(f"# Install {info['name']} ({info['size']}):")
                commands.append(f"ollama pull {model_id}")
                commands.append("")

        return "\n".join(commands)


# Singleton instance
_setup_checker = None


def get_setup_checker():
    global _setup_checker
    if _setup_checker is None:
        _setup_checker = SetupChecker()
    return _setup_checker
