"""
Dr. Data Tier 0 - Complete Foundation
Zero-Cloud AI for Solo Practitioners
"""

import hashlib
import json
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
import tempfile
import base64

import pandas as pd
import streamlit as st
import requests
import numpy as np

# Core modules
from core.audit_logger import TruthLinkLogger
from core.embeddings import LocalEmbedder
from core.governance import DeterministicGovernance
from core.llm_client import OllamaClient
from core.vector_store import LocalVectorStore
from core.config import OLLAMA_HOST, MODEL_NAME


# ============== SESSION STATE ==============

def init_session_state() -> None:
    """Initialize Streamlit session state."""
    defaults = {
        "messages": [],
        "user_id": str(uuid.uuid4()),
        "documents_loaded": False,
        "processed_file_ids": set(),
        "uploaded_data": None,  # Store DataFrame for analysis
        "current_file_name": None,
        "redaction_log": [],
        "embeddings_generated": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============== COMPONENT GETTERS ==============

def get_governance() -> DeterministicGovernance:
    if "governance" not in st.session_state:
        st.session_state.governance = DeterministicGovernance()
    return st.session_state.governance


def get_audit_logger() -> TruthLinkLogger:
    if "audit_logger" not in st.session_state:
        st.session_state.audit_logger = TruthLinkLogger()
    return st.session_state.audit_logger


def get_embedder() -> Optional[LocalEmbedder]:
    if "embedder" not in st.session_state:
        try:
            st.session_state.embedder = LocalEmbedder()
            st.session_state.embedder_error = None
        except Exception as e:
            st.session_state.embedder = None
            st.session_state.embedder_error = str(e)
    return st.session_state.embedder


def get_vector_store() -> Optional[LocalVectorStore]:
    if "vector_store" not in st.session_state:
        try:
            st.session_state.vector_store = LocalVectorStore()
        except Exception:
            st.session_state.vector_store = None
    return st.session_state.vector_store


def get_ollama_client() -> OllamaClient:
    if "ollama" not in st.session_state:
        st.session_state.ollama = OllamaClient()
    return st.session_state.ollama


# ============== FILE PROCESSING ==============

def process_uploaded_file(uploaded_file) -> tuple[Optional[str], Optional[pd.DataFrame]]:
    """
    Process uploaded file and return (text_content, dataframe).
    Returns DataFrame for CSV files, None for text files.
    """
    try:
        path = Path(uploaded_file.name)
        suffix = path.suffix.lower()
        
        if suffix == ".csv":
            df = pd.read_csv(uploaded_file)
            return df.to_string(), df
            
        elif suffix == ".txt":
            return uploaded_file.read().decode("utf-8", errors="replace"), None
            
        elif suffix == ".pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if not text.strip():
                    return "[Scanned PDF - no extractable text]", None
                return text, None
            except ImportError:
                st.error("PyPDF2 not installed. Run: pip install PyPDF2")
                return None, None
                
    except Exception as e:
        st.error(f"File processing error: {e}")
        return None, None


def process_image_file(image_file) -> Optional[str]:
    """
    Process image using Ollama vision model.
    Returns text description of the image.
    """
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image_file.name).suffix) as tmp:
            tmp.write(image_file.read())
            tmp_path = Path(tmp.name)
        
        # Read and encode image
        with open(tmp_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        
        # Clean up temp file
        tmp_path.unlink()
        
        # Call Ollama vision API
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": "llava:7b",
                "prompt": "Describe this image in detail. Extract any text, tables, charts, or data visible.",
                "images": [image_b64],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]
        
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to Ollama. Is it running?")
        return None
    except requests.exceptions.Timeout:
        st.error("Image analysis timed out. Try a smaller image.")
        return None
    except Exception as e:
        st.error(f"Image analysis failed: {e}")
        if "llava" in str(e).lower():
            st.info("Run: ollama pull llava:7b")
        return None


# ============== SIDEBAR ==============

def render_sidebar() -> None:
    """Render sidebar with system status."""
    st.sidebar.header("Dr. Data Tier 0")
    st.sidebar.caption("Zero-Cloud AI for Decision Intelligence")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    
    # Ollama status
    ollama = get_ollama_client()
    if ollama.is_available():
        st.sidebar.success("🟢 Ollama Phi-3: Ready")
    else:
        st.sidebar.error("🔴 Ollama: Not available")
    
    # Embeddings status
    embedder = get_embedder()
    if embedder:
        st.sidebar.success("🟢 Embeddings: Ready")
    else:
        st.sidebar.warning("🟡 Embeddings: Failed")
        if st.session_state.get("embedder_error"):
            with st.sidebar.expander("Error"):
                st.code(st.session_state.embedder_error[:200])
    
    # Vector store status
    vs = get_vector_store()
    if vs:
        st.sidebar.success("🟢 Vector Store: Ready")
    else:
        st.sidebar.warning("🟡 Vector Store: Failed")
    
    # Session info
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Session: {st.session_state.user_id[:8]}...")
    
    if st.session_state.documents_loaded:
        st.sidebar.info(f"📄 {len(st.session_state.processed_file_ids)} file(s) indexed")


# ============== TAB 1: HOME ==============

def render_home():
    """Render Home tab with overview."""
    st.header("🏠 Welcome to Dr. Data Tier 0")
    st.caption("Zero-Cloud AI for Decision Intelligence")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Files Indexed", len(st.session_state.processed_file_ids))
    with col2:
        st.metric("Redactions Made", len(st.session_state.redaction_log))
    with col3:
        st.metric("Chat Messages", len(st.session_state.messages))
    
    st.markdown("---")
    
    # Data Pipeline Visualization
    st.subheader("Data Pipeline")
    
    pipeline_steps = [
        ("📁 Upload", "Upload CSV, TXT, PDF, or Images"),
        ("🔒 Govern", "PII redaction & audit logging"),
        ("🔢 Embed", "Convert to vector representations"),
        ("💾 Store", "Index in vector database"),
        ("💬 Query", "Ask questions with RAG"),
    ]
    
    cols = st.columns(len(pipeline_steps))
    for i, (icon, desc) in enumerate(pipeline_steps):
        with cols[i]:
            st.markdown(f"**{icon}**")
            st.caption(desc)
    
    # Progress indicator
    progress = 0
    if st.session_state.processed_file_ids:
        progress += 1
    if st.session_state.redaction_log:
        progress += 1
    if st.session_state.embeddings_generated:
        progress += 1
    if st.session_state.documents_loaded:
        progress += 1
    if st.session_state.messages:
        progress += 1
    
    st.progress(progress / 5, text=f"Pipeline Progress: {progress}/5 steps")
    
    # Quick start
    st.markdown("---")
    st.subheader("Quick Start")
    st.markdown("""
    1. Go to **📁 Upload & Govern** tab to add your data
    2. Review **🔍 Data Quality** for insights
    3. Explore **📊 Embeddings Lab** to see how data is represented
    4. Use **💬 Query & RAG** to ask questions
    """)


# ============== TAB 2: UPLOAD & GOVERN ==============

def render_upload_govern():
    """Render Upload & Governance tab."""
    st.header("📁 Upload & Governance")
    
    col1, col2 = st.columns(2)
    
    # LEFT: File Uploads
    with col1:
        st.subheader("Documents")
        uploaded = st.file_uploader(
            "Upload CSV, TXT, or PDF",
            type=["csv", "txt", "pdf"],
            help="Documents will be redacted and indexed for RAG.",
        )
        
        if uploaded:
            file_id = (uploaded.name, uploaded.size)
            if file_id not in st.session_state.processed_file_ids:
                with st.spinner(f"Processing {uploaded.name}..."):
                    content, df = process_uploaded_file(uploaded)
                    
                    if content:
                        # Store DataFrame for analysis
                        if df is not None:
                            st.session_state.uploaded_data = df
                            st.session_state.current_file_name = uploaded.name
                        
                        # Apply governance
                        governance = get_governance()
                        audit = get_audit_logger()
                        
                        redacted, audit_trail = governance.redact_pii(content)
                        
                        # Log redactions
                        for entry in audit_trail:
                            audit.log_redaction({
                                "rule_id": entry.rule_id,
                                "position": entry.position,
                                "timestamp": entry.timestamp,
                                "original_length": entry.original_length,
                                "replacement": entry.replacement,
                            })
                            st.session_state.redaction_log.append(entry)
                        
                        # Index document
                        embedder = get_embedder()
                        vs = get_vector_store()
                        
                        if embedder and vs:
                            doc_id = f"doc_{uuid.uuid4().hex[:12]}"
                            emb = embedder.embed(redacted)
                            vs.add_document(
                                doc_id, 
                                redacted, 
                                {"source": uploaded.name, "type": "document"},
                                emb
                            )
                            st.session_state.documents_loaded = True
                            st.session_state.processed_file_ids.add(file_id)
                            st.session_state.embeddings_generated = True
                            st.success(f"✅ Indexed: {uploaded.name}")
                        else:
                            st.warning("⚠️ Embedder or vector store not ready.")
        
        st.markdown("---")
        st.subheader("📷 Images (Beta)")
        image_file = st.file_uploader(
            "Upload image for analysis",
            type=["png", "jpg", "jpeg"],
            key="image_uploader",
            help="Images are analyzed by AI to extract searchable text.",
        )
        
        if image_file:
            image_id = (image_file.name, image_file.size)
            if image_id not in st.session_state.processed_file_ids:
                with st.spinner(f"🔍 AI is analyzing {image_file.name}..."):
                    description = process_image_file(image_file)
                    
                    if description:
                        embedder = get_embedder()
                        vs = get_vector_store()
                        
                        if embedder and vs:
                            doc_id = f"img_{uuid.uuid4().hex[:12]}"
                            emb = embedder.embed(description)
                            vs.add_document(
                                doc_id,
                                description,
                                {
                                    "source": image_file.name,
                                    "type": "image_description",
                                    "original_filename": image_file.name,
                                },
                                emb
                            )
                            st.session_state.documents_loaded = True
                            st.session_state.processed_file_ids.add(image_id)
                            st.session_state.embeddings_generated = True
                            st.success(f"✅ Indexed image: {image_file.name}")
                            with st.expander("Preview description"):
                                st.write(description[:300] + "..." if len(description) > 300 else description)
    
    # RIGHT: Governance Preview
    with col2:
        st.subheader("🔒 Governance Preview")
        
        if st.session_state.uploaded_data is not None:
            df = st.session_state.uploaded_data
            
            st.markdown("**Data Preview**")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("**Redaction Summary**")
            if st.session_state.redaction_log:
                redaction_counts = {}
                for entry in st.session_state.redaction_log:
                    rule = entry.rule_id
                    redaction_counts[rule] = redaction_counts.get(rule, 0) + 1
                
                for rule, count in redaction_counts.items():
                    st.markdown(f"- {rule}: {count} instance(s)")
            else:
                st.caption("No PII detected in recent uploads.")
        else:
            st.info("Upload a file to see governance preview.")


# ============== TAB 3: DATA QUALITY ==============

def render_data_quality():
    """Render Data Quality tab."""
    st.header("🔍 Data Quality Lab")
    
    if st.session_state.uploaded_data is None:
        st.info("📁 Upload a CSV file in the 'Upload & Govern' tab to see data quality analysis.")
        return
    
    df = st.session_state.uploaded_data
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        missing = df.isnull().sum().sum()
        st.metric("Missing Values", missing)
    with col4:
        duplicates = df.duplicated().sum()
        st.metric("Duplicate Rows", duplicates)
    
    st.markdown("---")
    
    # Column analysis
    st.subheader("Column Analysis")
    
    col_analysis = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_pct = df[col].isnull().mean() * 100
        unique = df[col].nunique()
        
        col_analysis.append({
            "Column": col,
            "Type": dtype,
            "Missing %": f"{missing_pct:.1f}%",
            "Unique Values": unique,
        })
    
    st.dataframe(pd.DataFrame(col_analysis), use_container_width=True)
    
    # Data preview
    st.markdown("---")
    st.subheader("Data Preview")
    st.dataframe(df, use_container_width=True)
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations")
    
    recommendations = []
    
    if duplicates > 0:
        recommendations.append(f"🔴 Found {duplicates} duplicate rows. Consider removing them.")
    
    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct > 50:
            recommendations.append(f"🟡 Column '{col}' has {missing_pct:.0f}% missing values. Consider dropping it.")
        elif missing_pct > 0:
            recommendations.append(f"🟢 Column '{col}' has {missing_pct:.0f}% missing values.")
    
    if not recommendations:
        st.success("✅ Data looks clean! No major issues detected.")
    else:
        for rec in recommendations:
            st.markdown(rec)


# ============== TAB 4: EMBEDDINGS LAB ==============

def render_embeddings_lab():
    """Render Embeddings Lab tab."""
    st.header("📊 Embeddings Lab")
    
    if not st.session_state.documents_loaded:
        st.info("📁 Upload and index documents to explore embeddings.")
        return
    
    st.subheader("Embedding Status")
    
    embedder = get_embedder()
    if embedder:
        st.success(f"✅ Model: {embedder.MODEL_NAME}")
        st.caption(f"Dimension: {embedder.DIMENSION}d vectors")
    
    st.markdown("---")
    
    # Document statistics
    st.subheader("Document Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Indexed Documents", len(st.session_state.processed_file_ids))
    with col2:
        st.metric("Redactions Applied", len(st.session_state.redaction_log))
    
    # Placeholder for future visualization
    st.markdown("---")
    st.subheader("🔮 Vector Visualization")
    st.info("2D/3D visualization of document clusters coming soon!")
    
    st.caption("Embeddings convert your documents into mathematical vectors that capture semantic meaning.")


# ============== TAB 5: QUERY & RAG ==============

def render_query_rag():
    """Render Query & RAG tab."""
    st.header("💬 Query Your Data")
    
    # Chat interface
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "provenance" in msg:
                with st.expander("📜 Truth-Link Provenance"):
                    st.json(msg["provenance"])
    
    # Query input
    if prompt := st.chat_input("Ask about your data..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                governance = get_governance()
                audit = get_audit_logger()
                ollama = get_ollama_client()
                embedder = get_embedder()
                vs = get_vector_store()
                
                # Redact query
                redacted_prompt, trail = governance.redact_pii(prompt)
                
                # Get context from vector store
                context_parts = []
                source_ids = []
                
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
                    except Exception as e:
                        st.warning(f"RAG lookup issue: {e}")
                
                context = "\n\n".join(context_parts) if context_parts else None
                
                # Generate response
                try:
                    response = ollama.chat(redacted_prompt, context)
                except Exception as e:
                    response = f"Error: {e}"
                
                # Log to audit
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
                    st.warning(f"Audit log issue: {e}")
                
                # Display response
                st.markdown(response)
                
                # Provenance
                provenance = {
                    "response_hash": response_hash,
                    "sources": context_parts[:3] if context_parts else [],
                    "redactions": len(trail),
                }
                
                with st.expander("📜 Truth-Link Provenance"):
                    st.json(provenance)
        
        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "provenance": provenance,
        })


# ============== MAIN ==============

def main():
    st.set_page_config(
        page_title="Dr. Data Tier 0",
        page_icon="📊",
        layout="wide",
    )
    
    init_session_state()
    render_sidebar()
    
    # Create tabs
    tabs = st.tabs([
        "🏠 Home",
        "📁 Upload & Govern",
        "🔍 Data Quality",
        "📊 Embeddings Lab",
        "💬 Query & RAG",
    ])
    
    with tabs[0]:
        render_home()
    
    with tabs[1]:
        render_upload_govern()
    
    with tabs[2]:
        render_data_quality()
    
    with tabs[3]:
        render_embeddings_lab()
    
    with tabs[4]:
        render_query_rag()


if __name__ == "__main__":
    main()
