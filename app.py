"""
Dr. Data Tier 0 - Streamlit entry point.
Zero-Cloud AI for solo practitioners.
"""

import hashlib
import uuid
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

from core.audit_logger import TruthLinkLogger
from core.embeddings import LocalEmbedder
from core.governance import DeterministicGovernance
from core.llm_client import OllamaClient
from core.vector_store import LocalVectorStore


def init_session_state() -> None:
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "processed_file_ids" not in st.session_state:
        st.session_state.processed_file_ids = set()


def get_governance() -> DeterministicGovernance:
    """Get or create governance instance."""
    if "governance" not in st.session_state:
        st.session_state.governance = DeterministicGovernance()
    return st.session_state.governance


def get_audit_logger() -> TruthLinkLogger:
    """Get or create audit logger instance."""
    if "audit_logger" not in st.session_state:
        st.session_state.audit_logger = TruthLinkLogger()
    return st.session_state.audit_logger


def get_embedder() -> Optional[LocalEmbedder]:
    """Get or create embedder (lazy load)."""
    if "embedder" not in st.session_state:
        try:
            st.session_state.embedder = LocalEmbedder()
        except Exception:
            st.session_state.embedder = None
    return st.session_state.embedder


def get_vector_store() -> Optional[LocalVectorStore]:
    """Get or create vector store (lazy load)."""
    if "vector_store" not in st.session_state:
        try:
            st.session_state.vector_store = LocalVectorStore()
        except Exception:
            st.session_state.vector_store = None
    return st.session_state.vector_store


def get_ollama_client() -> OllamaClient:
    """Get or create Ollama client."""
    if "ollama" not in st.session_state:
        st.session_state.ollama = OllamaClient()
    return st.session_state.ollama


def process_uploaded_file(uploaded_file) -> Optional[str]:
    """
    Process uploaded CSV or TXT file and return content string.

    Returns:
        File content or None on error.
    """
    try:
        path = Path(uploaded_file.name)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(uploaded_file)
            return df.to_string()
        if suffix == ".txt":
            return uploaded_file.read().decode("utf-8", errors="replace")
    except Exception as e:
        st.error(f"File processing error: {e}")
    return None


def render_sidebar() -> None:
    """Render sidebar with file upload and system status."""
    st.sidebar.header("Dr. Data Tier 0")
    st.sidebar.subheader("File Upload")
    uploaded = st.sidebar.file_uploader(
        "Upload CSV or TXT",
        type=["csv", "txt"],
        help="Documents will be redacted and indexed for RAG.",
    )

    if uploaded:
        file_id = (uploaded.name, uploaded.size)
        if file_id not in st.session_state.processed_file_ids:
            content = process_uploaded_file(uploaded)
            if content:
                governance = get_governance()
                audit = get_audit_logger()
                redacted, audit_trail = governance.redact_pii(content)
                for entry in audit_trail:
                    audit.log_redaction(
                        {
                            "rule_id": entry.rule_id,
                            "position": entry.position,
                            "timestamp": entry.timestamp,
                            "original_length": entry.original_length,
                            "replacement": entry.replacement,
                        }
                    )
                embedder = get_embedder()
                vs = get_vector_store()
                if embedder and vs:
                    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
                    emb = embedder.embed(redacted)
                    vs.add_document(doc_id, redacted, {"source": uploaded.name}, emb)
                    st.session_state.documents_loaded = True
                    st.session_state.processed_file_ids.add(file_id)
                    st.sidebar.success(f"Indexed: {uploaded.name}")
                else:
                    st.sidebar.warning("Embedder or vector store not ready.")

    st.sidebar.subheader("System Status")
    ollama = get_ollama_client()
    if ollama.is_available():
        st.sidebar.success("Ollama Phi-3: Ready")
    else:
        st.sidebar.error("Ollama: Not available (start Ollama and pull phi3:mini)")
    embedder = get_embedder()
    if embedder:
        st.sidebar.success("Embeddings: Ready")
    else:
        st.sidebar.warning("Embeddings: Failed to load")


def main() -> None:
    """Main Streamlit app."""
    st.set_page_config(page_title="Dr. Data Tier 0", page_icon="📊", layout="wide")
    st.title("Dr. Data Tier 0")
    st.caption("Zero-Cloud AI for Decision Intelligence")

    init_session_state()
    render_sidebar()

    governance = get_governance()
    audit = get_audit_logger()
    ollama = get_ollama_client()
    embedder = get_embedder()
    vs = get_vector_store()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "provenance" in msg:
                with st.expander("Truth-Link Provenance"):
                    st.json(msg["provenance"])

    if prompt := st.chat_input("Ask about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        redacted_prompt, trail = governance.redact_pii(prompt)
        for entry in trail:
            audit.log_redaction(
                {
                    "rule_id": entry.rule_id,
                    "position": entry.position,
                    "timestamp": entry.timestamp,
                    "original_length": entry.original_length,
                    "replacement": entry.replacement,
                }
            )

        with st.chat_message("user"):
            st.markdown(prompt)
            if trail:
                st.caption("Redacted before AI: " + redacted_prompt[:100] + ("..." if len(redacted_prompt) > 100 else ""))

        with st.chat_message("assistant"):
            context_parts: List[str] = []
            source_ids: List[str] = []
            if vs and embedder:
                try:
                    query_emb = embedder.embed(redacted_prompt)
                    results = vs.query(query_embedding=query_emb, n_results=3)
                    for r in results:
                        text = r.get("document") or r.get("metadata", {}).get("text", "")
                        if text:
                            context_parts.append(text[:500])
                        if r.get("id"):
                            source_ids.append(r["id"])
                    context = "\n\n".join(context_parts) if context_parts else None
                except Exception as e:
                    st.warning(f"RAG lookup failed: {e}")
                    context = None
            else:
                context = None

            try:
                response = ollama.chat(redacted_prompt, context)
            except Exception as e:
                response = f"Error: {e}"

            response_hash = hashlib.sha256(response.encode()).hexdigest()
            try:
                audit.log_query(
                    user_id=st.session_state.user_id,
                    query=redacted_prompt,
                    sources=source_ids,
                    response_hash=response_hash,
                    metadata={"redaction_count": len(trail)},
                )
            except Exception as e:
                st.warning(f"Audit log failed: {e}")

            provenance = {
                "response_hash": response_hash,
                "sources": context_parts[:3],
                "redactions": len(trail),
            }

            st.markdown(response)
            with st.expander("Truth-Link Provenance"):
                st.json(provenance)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "provenance": provenance,
            }
        )


if __name__ == "__main__":
    main()
