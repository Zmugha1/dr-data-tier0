"""
Dr Data - Tier 0 RAG/GraphRAG Lab
Deterministic, Idempotent Document Processing Pipeline
"""

import json
import shutil
import time
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Dr Data - Tier 0 RAG/GraphRAG Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

# MUST BE FIRST: Handle reset before any ChromaDB imports
if "reset_triggered" not in st.session_state:
    st.session_state.reset_triggered = False
    st.session_state.reset_success = False
    st.session_state.reset_scope = "vector"  # "vector" | "full"
    st.session_state.sidebar_reset_confirm = False

# Check if we need to reset (from previous run) - runs BEFORE VectorStore import
if st.session_state.reset_triggered and not st.session_state.reset_success:
    from core.reset_utils import reset_vector_db_nuclear, reset_all_data_nuclear

    if st.session_state.reset_scope == "full":
        success, msg = reset_all_data_nuclear()
    else:
        success, msg = reset_vector_db_nuclear()

    if success:
        st.session_state.reset_success = True
        st.session_state.reset_triggered = False
        st.session_state.reset_scope = "vector"
        st.success(f"✅ {msg}")
        st.info("Reloading app...")
        st.rerun()
    else:
        st.error(f"❌ Reset failed: {msg}")
        _fail_path = str(Path(__file__).resolve().parent / "data" / "vector_db")
        st.code(f"""
Manual fix (run in terminal):
taskkill /f /im python.exe 2>nul
rd /s /q "{_fail_path}"
streamlit run app.py
""", language="batch")
        st.stop()

# Create data directories
for dir_path in ["data/raw", "data/vector_db", "data/graph_store", "data/audit_logs"]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# NOW import the rest (after potential reset)
from core.vector_store import VectorStore

# Initialize VectorStore with error handling
vector_store = None
init_error = None

try:
    vector_store = VectorStore()
    vector_store.initialize(reset=False)
except Exception as e:
    error_str = str(e).lower()
    if "different settings" in error_str or "already exists" in error_str:
        init_error = "settings_conflict"
    else:
        init_error = "other"
        st.error(f"Initialization error: {e}")

# UI for conflict resolution
if init_error == "settings_conflict":
    st.error("⚠️ Vector DB Settings Conflict")

    st.warning("""
The Vector DB was created with a different embedding model.

**You must reset to continue.** The buttons below will force-delete the corrupted database.
""")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ FIX: Reset Vector DB", type="primary", use_container_width=True):
            st.session_state.reset_triggered = True
            st.session_state.reset_success = False
            st.session_state.reset_scope = "vector"
            st.rerun()

    with col2:
        if st.button("☢️ Full Reset", use_container_width=True):
            st.session_state.reset_triggered = True
            st.session_state.reset_success = False
            st.session_state.reset_scope = "full"
            st.rerun()

    st.divider()
    st.subheader("If buttons don't work (Windows File Lock):")
    _db_path = str(Path(__file__).resolve().parent / "data" / "vector_db")
    st.code(f"""
1. Close this Streamlit window (Ctrl+C in terminal)
2. Open Command Prompt as Administrator
3. Run:
   taskkill /f /im python.exe
   rd /s /q "{_db_path}"
4. cd to your project folder, then: streamlit run app.py
""", language="batch")

    st.stop()

elif init_error == "other":
    st.error("Fatal error initializing storage. Check logs.")
    if st.button("Retry"):
        st.rerun()
    st.stop()

elif vector_store is None:
    st.error("Vector Store failed to initialize.")
    st.stop()

st.title("🏥 Dr Data - Zero-Cloud AI Architecture Lab")
st.markdown("""
**What this page does**: This is the main hub for ingesting documents into your RAG/GraphRAG pipeline.  
Upload PDFs, CSVs, TXT, or Excel files to build both a **Vector DB** (semantic search) and **Knowledge Graph** (entity relationships) at once.  
All processing is deterministic and idempotent—duplicate files are automatically skipped.
""")

# Sidebar Status
with st.sidebar:
    st.header("System Status")
    st.caption("Shows readiness of Ollama, Vector DB, and Knowledge Graph. Process documents to populate.")

    vec_stats = vector_store.get_stats() if vector_store else {}
    if vec_stats.get("status") == "Active":
        st.success(f"🟢 Vector DB: {vec_stats.get('chunks', 0)} chunks")
    else:
        st.warning("🟡 Vector DB Empty")

    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = [m["name"] for m in response.json()["models"]]
            st.success(f"🟢 Ollama Ready\n{len(models)} models cached")
            st.caption(f"Available: {', '.join(models[:3])}...")
    except Exception:
        st.error("🔴 Ollama Offline\nRun: `ollama serve`")

    raw_count = len(list(Path("data/raw").glob("**/*.*")))
    st.info(f"📁 Documents Stored: {raw_count}")

    if Path("data/graph_store/knowledge_graph.pkl").exists():
        st.success("🟢 Knowledge Graph Ready")
    else:
        st.warning("🟡 Knowledge Graph Empty")

    st.divider()
    st.caption("🛠️ Maintenance")
    if st.button("🔄 Reset Vector DB", help="Clear embeddings (keeps documents)"):
        st.session_state.reset_triggered = True
        st.session_state.reset_success = False
        st.session_state.reset_scope = "vector"
        st.rerun()
    with st.expander("Advanced"):
        if st.button("☢️ Factory Reset", help="Clear everything"):
            st.session_state.sidebar_reset_confirm = True
        if st.session_state.get("sidebar_reset_confirm"):
            if st.checkbox("Confirm delete all data", key="confirm_factory"):
                if st.button("Execute Factory Reset"):
                    st.session_state.reset_triggered = True
                    st.session_state.reset_success = False
                    st.session_state.reset_scope = "full"
                    st.session_state.sidebar_reset_confirm = False
                    st.rerun()

# Main Tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Pipeline Status", "🔍 Data Explorer"])

with tab1:
    st.header("Phase 1: Document Ingestion & Canonicalization")

    st.info("""
    **How to use this tab**:  
    1. **Upload** — Click *Browse files* or drag-and-drop documents (PDF, CSV, TXT, XLSX).  
    2. **Optional** — Adjust chunk size/overlap in the expander below if needed.  
    3. **Process** — Click **Process Documents** to run the pipeline. First run may take a minute (embedding model loads).  
    4. **Result** — A table shows each file's status (Processed or Skipped if duplicate).  
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Drop documents (PDF, CSV, TXT, XLSX)",
            type=["pdf", "csv", "txt", "xlsx"],
            accept_multiple_files=True,
            help="Files are hashed (SHA-256) before processing. Duplicates are automatically skipped.",
        )

        processing_options = st.expander("⚙️ Deterministic Processing Options")
        with processing_options:
            st.caption("Adjust these only if you need different chunk sizes. Defaults work for most documents.")
            chunk_size = st.slider("Chunk Size (tokens)", 256, 1024, 512, 64)
            chunk_overlap = st.slider("Chunk Overlap", 0, 100, 50, 10)
            ocr_enabled = st.checkbox("Enable OCR for scanned PDFs", value=True)
            st.caption("Using Tesseract with fixed DPI=300, PSM=6")

    with col2:
        st.markdown("""
        **Idempotency Check:**
        - Files hashed via SHA-256
        - Duplicate detection active
        - Version-pinned embeddings
        - Audit trail logging
        """)
        st.caption("Click the button below after selecting files.")
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("Upload files first")
            elif vector_store is None:
                st.error("Vector Store not available. Reset the database using the button in the sidebar.")
                st.stop()
            else:
                with st.spinner("Running 19-step deterministic pipeline..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    from core.audit_logger import AuditLogger
                    from core.document_processor import DocumentProcessor
                    from core.knowledge_graph import KnowledgeGraphBuilder

                    logger = AuditLogger()
                    processor = DocumentProcessor(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )
                    kg_builder = KnowledgeGraphBuilder()

                    results = []
                    for idx, file in enumerate(uploaded_files):
                        status_text.text(f"Processing {file.name}...")

                        file_hash = processor.compute_hash(file.getvalue())

                        if processor.hash_exists(file_hash):
                            results.append(
                                {
                                    "file": file.name,
                                    "status": "Skipped (duplicate)",
                                    "hash": file_hash[:8],
                                }
                            )
                            continue

                        doc_data = processor.process(file, file_hash, ocr_enabled)

                        vector_store.add_document(doc_data)
                        kg_builder.add_document(doc_data)
                        logger.log_ingestion(file.name, file_hash, doc_data["chunks"])

                        results.append(
                            {
                                "file": file.name,
                                "status": "Processed",
                                "hash": file_hash[:8],
                                "chunks": len(doc_data["chunks"]),
                                "entities": len(
                                    doc_data.get("entities", [])
                                ),
                            }
                        )
                        progress_bar.progress((idx + 1) / len(uploaded_files))

                    kg_stats = kg_builder.persist()
                    vector_store.persist()
                    logger.finalize_batch(
                        doc_count=len([r for r in results if r["status"] == "Processed"]),
                        total_chunks=sum(r.get("chunks", 0) for r in results),
                        total_entities=kg_stats.get("nodes", 0),
                        total_relations=kg_stats.get("edges", 0),
                    )

                    processed = len([r for r in results if r["status"] == "Processed"])
                    st.success(f"✅ Processed {processed} new documents")
                    st.dataframe(pd.DataFrame(results), use_container_width=True)

with tab2:
    st.header("Pipeline Execution Status")
    st.caption("View metrics and workflow status after processing documents. No action required—data updates automatically.")

    if Path("data/audit_logs/latest_manifest.json").exists():
        with open("data/audit_logs/latest_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Documents", manifest.get("doc_count", 0))
        col2.metric("Total Chunks", manifest.get("total_chunks", 0))
        col3.metric("Entities Extracted", manifest.get("total_entities", 0))
        col4.metric("Relations", manifest.get("total_relations", 0))

        st.subheader("19-Step Workflow Progress")
        steps = [
            ("✅ Phase 1: Ingestion", "Content-addressable storage, deterministic extraction"),
            ("✅ Phase 2: Vector DB", "Nomic-embed-text, ChromaDB persistence"),
            ("✅ Phase 3: Knowledge Graph", "Phi-4 entity extraction, NetworkX storage"),
            ("⏳ Phase 4: Tabular Integration", "Awaiting CSV upload with foreign keys"),
            ("⏳ Phase 5: ML Features", "PageRank, clustering (on demand)"),
            ("⏳ Phase 6: LLM Inference", "Ready for chat queries"),
            ("✅ Phase 7: Audit Trail", f"Execution hash: {manifest.get('batch_hash', 'N/A')[:16]}..."),
        ]

        for step, desc in steps:
            st.markdown(f"**{step}** - {desc}")
    else:
        st.info("No documents processed yet. Upload files to see pipeline status.")

with tab3:
    st.header("Data Structure Explorer")
    st.caption("Inspect the knowledge graph built from your documents. Shows node counts, sample entities, and graph density. No interaction needed—just browse the stats.")

    if Path("data/graph_store/knowledge_graph.pkl").exists():
        import pickle

        import networkx as nx

        with open("data/graph_store/knowledge_graph.pkl", "rb") as f:
            G = pickle.load(f)

        st.subheader("Knowledge Graph Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Nodes", G.number_of_nodes())
        col2.metric("Edges", G.number_of_edges())
        col3.metric("Density", f"{nx.density(G):.4f}")

        st.subheader("Sample Entities")
        nodes_df = pd.DataFrame(
            [
                {
                    "id": n,
                    "type": G.nodes[n].get("type", "unknown"),
                    "connections": G.degree(n),
                }
                for n in list(G.nodes())[:10]
            ]
        )
        st.dataframe(nodes_df, use_container_width=True)
    else:
        st.info("Process documents to build knowledge graph")

st.markdown("---")
st.caption("Dr Data Tier 0 - Deterministic, Idempotent, Air-Gapped")
