#!/usr/bin/env python3
"""
Quick check script to verify Tier 0 setup
"""

import sys


def check() -> int:
    """Run all setup checks and return exit code."""
    print("🔍 Dr. Data Tier 0 - Quick Check\n")

    checks = []

    import platform

    version = platform.python_version()
    py_ok = version.startswith("3.11")
    checks.append(("Python 3.11", py_ok, version))

    try:
        import sentence_transformers

        st_ok = True
        st_ver = getattr(sentence_transformers, "__version__", "installed")
    except ImportError:
        st_ok = False
        st_ver = "not installed"
    checks.append(("sentence_transformers", st_ok, st_ver))

    try:
        import streamlit

        stl_ok = True
        stl_ver = getattr(streamlit, "__version__", "installed")
    except ImportError:
        stl_ok = False
        stl_ver = "not installed"
    checks.append(("streamlit", stl_ok, stl_ver))

    try:
        import chromadb

        ch_ok = True
        ch_ver = getattr(chromadb, "__version__", "installed")
    except ImportError:
        ch_ok = False
        ch_ver = "not installed"
    checks.append(("chromadb", ch_ok, ch_ver))

    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        ollama_ok = response.status_code == 200
        ollama_ver = "running" if ollama_ok else "not responding"
    except Exception:
        ollama_ok = False
        ollama_ver = "not running"
    checks.append(("Ollama", ollama_ok, ollama_ver))

    all_ok = True
    for name, ok, info in checks:
        status = "✅" if ok else "❌"
        print(f"{status} {name}: {info}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("🎉 All checks passed! Ready to run: streamlit run app.py")
        return 0
    else:
        print("⚠️  Some checks failed. Run: py setup.py")
        return 1


if __name__ == "__main__":
    sys.exit(check())
