# Dr. Data Tier 0 - Edge Micro-Hub

**Zero-Cloud AI for Solo Practitioners**

Dr. Data Tier 0 is a local-first, privacy-preserving decision intelligence system designed for individual practitioners who require AI assistance without sending sensitive data to the cloud. All processing runs entirely on your machine using Ollama and local embeddings.

## Features

- **Deterministic PII Redaction** — Regex-based governance (no AI) for SSN, phone, email, credit card
- **Truth-Link Audit Trail** — Immutable append-only JSONL logs for compliance
- **Local Embeddings** — BGE-Micro-v2 (384-dim) via sentence-transformers
- **ChromaDB Vector Store** — Persistent document indexing
- **Ollama Phi-3** — Local LLM inference via HTTP API

## Quick Start

1. **Install Ollama** and pull the Phi-3 model:
   ```bash
   ollama pull phi3:mini
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. Open http://localhost:8501 in your browser.

## Hardware Requirements

- **RAM:** 8GB minimum; 16GB recommended for Phi-3 + embeddings
- **Storage:** ~4GB for model artifacts (Ollama Phi-3 + BGE-Micro)
- **CPU:** Multi-core recommended
- **GPU:** Optional; Ollama supports CUDA/Metal for faster inference

## Project Structure

```
dr-data-tier0/
├── app.py              # Streamlit entry point
├── core/
│   ├── config.py       # PII patterns, HIPAA keywords, settings
│   ├── governance.py   # Deterministic PII redaction
│   ├── audit_logger.py # Truth-Link immutable logs
│   ├── embeddings.py   # BGE-Micro local embedder
│   ├── vector_store.py # ChromaDB wrapper
│   └── llm_client.py   # Ollama Phi-3 interface
└── requirements.txt
```

## Audit Logs

Audit logs are stored in `logs/` as append-only JSONL. Each redaction and query is logged with timestamps and provenance for compliance. Push and commit logs to [dr-data-tier0](https://github.com/Zmugha1/dr-data-tier0) for version control and external audit.

## License

MIT
