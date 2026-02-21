#!/usr/bin/env python3
"""
Setup script for Dr. Data Tier 0
Run this first to install all dependencies
"""

import subprocess
import sys

def run_command(cmd: str, description: str) -> bool:
    """Run shell command with error handling."""
    print(f"\n{'='*50}")
    print(f"[*] {description}")
    print(f"{'='*50}")
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=False, text=True)
        print(f"[OK] {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {description} failed: {e}")
        return False


def main() -> None:
    """Run full setup: install deps, pre-download model, check Ollama."""
    print("Dr. Data Tier 0 - Setup")
    print("=" * 50)

    import platform

    version = platform.python_version()
    print(f"Python version: {version}")

    if not version.startswith("3.11"):
        print("[!] Warning: Python 3.11.x recommended")
        print(f"   Current: {version}")

    if not run_command("py -m pip install -r requirements.txt", "Installing dependencies"):
        sys.exit(1)

    print(f"\n{'='*50}")
    print("[*] Pre-downloading BGE-Micro model (30MB)")
    print(f"{'='*50}")
    print("This may take 2-5 minutes depending on internet...")

    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer("TaylorAI/bge-micro-v2")
        print("[OK] BGE-Micro model downloaded and cached")
    except Exception as e:
        print(f"[FAIL] Model download failed: {e}")
        print("The app will download on first run instead")

    print(f"\n{'='*50}")
    print("[*] Checking Ollama")
    print(f"{'='*50}")
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("[OK] Ollama is running")
            models = response.json().get("models", [])
            phi3_found = any("phi3" in m.get("name", "").lower() for m in models)
            if phi3_found:
                print("[OK] Phi-3 Mini model found")
            else:
                print("[!] Phi-3 Mini not found. Run: ollama pull phi3:mini")
        else:
            print("[FAIL] Ollama not responding")
    except Exception:
        print("[FAIL] Ollama not running. Start with: ollama serve")

    print(f"\n{'='*50}")
    print("[OK] Setup complete!")
    print(f"{'='*50}")
    print("\nNext steps:")
    print("1. Make sure Ollama is running: ollama serve")
    print("2. Run the app: streamlit run app.py")


if __name__ == "__main__":
    main()
