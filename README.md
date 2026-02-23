# Dr Data Core Architecture Planning

**Zero-Cloud AI for Solo Practitioners**

Dr Data Core Architecture Planning is a local-first, privacy-preserving decision intelligence system designed for individual practitioners who require AI assistance without sending sensitive data to the cloud. All processing runs entirely on your machine using Ollama and local embeddings.

## Features

- **Deterministic PII Redaction** — Regex-based governance (no AI) for SSN, phone, email, credit card
- **Truth-Link Audit Trail** — Immutable append-only JSONL logs for compliance
- **Local Embeddings** — BGE-Micro-v2 (384-dim) via sentence-transformers
- **ChromaDB Vector Store** — Persistent document indexing
- **Ollama Phi-3** — Local LLM inference via HTTP API

## Quick Setup (Windows)

### Option 1: Automated Setup

```powershell
# 1. Navigate to project
cd C:\Users\zumah\Documents\dr-data-tier0

# 2. Run setup script (installs deps + downloads model)
py setup.py

# 3. Verify installation
py check_setup.py

# 4. Run the app
streamlit run app.py
```

### Option 2: Manual Setup

```powershell
# 1. Create virtual environment
py -m venv venv

# 2. Activate it
.\venv\Scripts\activate

# 3. Install dependencies
py -m pip install -r requirements.txt

# 4. Pre-download model (optional, speeds up first run)
py -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-micro-v2')"

# 5. Run
streamlit run app.py
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'sentence_transformers'` | Run: `py setup.py` or `py -m pip install -r requirements.txt` |
| App stuck on "Loading BGE-Micro model" | Model is downloading (30MB). Wait 2-5 minutes or run `py setup.py` first. |
| "Ollama connection refused" | Start Ollama: `ollama serve` (in separate terminal) |

**Install Ollama** and pull Phi-3 before running: `ollama pull phi3:mini`

Open http://localhost:8501 in your browser.

## Hardware Requirements

- **RAM:** 8GB minimum; 16GB recommended for Phi-3 + embeddings
- **Storage:** ~4GB for model artifacts (Ollama Phi-3 + BGE-Micro)
- **CPU:** Multi-core recommended
- **GPU:** Optional; Ollama supports CUDA/Metal for faster inference

## Project Structure

```
dr-data-tier0/
├── app.py              # Streamlit entry point
├── streamlit_app.py    # Streamlit Cloud entry point
├── setup.py            # Install deps + pre-download model
├── check_setup.py      # Verify installation
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
