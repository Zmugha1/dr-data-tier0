"""
Configuration constants and patterns for Dr. Data Tier 0.
"""

from typing import Dict, List

# PII regex patterns with word boundaries
PII_PATTERNS: Dict[str, str] = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "credit_card": r"\b(?:\d{4}[-.\s]?){3}\d{4}\b",
}

# HIPAA-sensitive keywords for awareness (not used in regex redaction but for audit context)
HIPAA_KEYWORDS: List[str] = [
    "patient",
    "diagnosis",
    "prescription",
    "medical record",
    "health record",
    "PHI",
    "protected health information",
    "treatment",
    "provider",
    "DOB",
    "date of birth",
]

# Ollama API settings
OLLAMA_HOST: str = "http://localhost:11434"
MODEL_NAME: str = "phi3:mini"

# ChromaDB persistence path
CHROMA_PATH: str = "./chroma_db"
