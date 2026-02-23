"""Architecture Viewer - 19-step workflow, data structures, determinism proof."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Architecture Viewer", layout="wide")

st.title("🏗️ Dr Data Architecture Visualizer")
st.markdown("""
**What this page does**: Inspect the 19-step deterministic pipeline, data schemas, audit hash chain, and RAG vs GraphRAG comparison.  
**Read-only** — no uploads or actions. Use the tabs below to explore different aspects of the architecture.
""")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Pipeline Flow", "Data Structures", "Determinism Proof", "Comparison"]
)

with tab1:
    st.header("19-Step Deterministic Workflow")
    st.caption("Each step is executed in order during document processing. Green checkmarks indicate completed phases.")

    steps = [
        ("1. Hash (SHA-256)", "Content-addressable storage", "🔐"),
        ("2. Extract", "pdfplumber / Tesseract OCR", "📄"),
        ("3. Chunk", "512 tokens, 50 overlap", "✂️"),
        ("4. Embed", "Nomic-text-v1.5", "🧮"),
        ("5. Index", "ChromaDB (cosine)", "🗄️"),
        ("6. Entity Extract", "Phi-4 (temp=0)", "🎯"),
        ("7. Canonicalize", "Fuzzy matching (0.85)", "🔗"),
        ("8. Graph Store", "NetworkX / Neo4j", "🕸️"),
        ("9. Schema Align", "ISO-8601 dates", "📊"),
        ("10. Graph Enrich", "Tabular attributes", "💎"),
        ("11. Features", "PageRank (100 iter)", "⚙️"),
        ("12. Cluster", "K-Means (seed=42)", "🎨"),
        ("13. Predict", "XGBoost (seed=42)", "🔮"),
        ("14. Context Assembly", "BFS 2-hop + Top-k", "🔍"),
        ("15. Generate", "Qwen 14B (temp=0)", "💬"),
        ("16. Validate", "3-sigma rule", "✅"),
        ("17. Hash Chain", "SHA256(prev+curr)", "⛓️"),
        ("18. Version Pin", "Model manifest", "📌"),
        ("19. Export", "Parquet + TTL", "📦"),
    ]

    cols = st.columns(4)
    for idx, (step, desc, icon) in enumerate(steps):
        with cols[idx % 4]:
            with st.container(border=True):
                st.markdown(f"**{icon} {step}**")
                st.caption(desc)

                vec_ready = Path("data/vector_db/chroma.sqlite3").exists()
                vec_manifest = Path("data/vector_db/vector_manifest.json").exists()
                graph_ready = Path("data/graph_store/knowledge_graph.pkl").exists()

                if step in ("4. Embed", "5. Index") and (vec_ready or vec_manifest):
                    st.success("✓ Complete")
                elif step == "8. Graph Store" and graph_ready:
                    st.success("✓ Complete")

with tab2:
    st.header("Data Structure Inspector")
    st.caption("View the schema and current state of your Vector DB and Knowledge Graph (after processing documents).")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Vector DB Schema")
        st.code(
            """
Collection: dr_data_default
├── id: content_hash (SHA-256)
├── document: text_chunk
├── metadata:
│   ├── doc_hash: str
│   ├── chunk_index: int
│   └── source: filename
└── embedding: float[768]  # Nomic-embed

Distance Metric: Cosine Similarity
        """,
            language="yaml",
        )

        if Path("data/vector_db/vector_manifest.json").exists():
            with open("data/vector_db/vector_manifest.json", encoding="utf-8") as f:
                vm = json.load(f)
            st.json(vm)

    with col2:
        st.subheader("Knowledge Graph Schema")
        st.code(
            """
Node Types:
- Document {hash, filename, timestamp}
- Entity {type, name, canonical_id}
- Chunk {index, text_hash}

Edge Types:
- CONTAINS (Doc → Chunk)
- MENTIONS (Chunk → Entity)
- RELATION (Entity → Entity)
  [purchased_from, amount_of, dated, ...]

Storage: NetworkX DiGraph (pickled)
Backup: GEXF (Cytoscape compatible)
        """,
            language="yaml",
        )

        if Path("data/graph_store/knowledge_graph.pkl").exists():
            import pickle

            with open("data/graph_store/knowledge_graph.pkl", "rb") as f:
                G = pickle.load(f)
            st.metric("Total Nodes", G.number_of_nodes())
            st.metric("Total Edges", G.number_of_edges())

with tab3:
    st.header("Determinism & Idempotency Proof")
    st.caption("Audit chain proving that reprocessing the same file produces no duplicates and preserves traceability.")

    st.markdown("""
    ### Idempotency Check

    If you upload the same file twice, the system:
    1. Computes SHA-256 hash
    2. Checks `/data/raw/{prefix}/{hash}.json`
    3. Skips processing if exists
    4. Returns existing vector/graph references

    **Result**: Zero duplicate storage, zero duplicate processing cost.
    """)

    if Path("data/audit_logs/latest_manifest.json").exists():
        with open("data/audit_logs/latest_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)

        st.subheader("Current Audit Chain")
        st.json(manifest, expanded=False)

        entries = manifest.get("entries", [])
        if entries:
            st.markdown("**Hash Chain (Last 3)**")
            for i, entry in enumerate(entries[-3:]):
                cols = st.columns([1, 3, 2])
                cols[0].write(f"Block {i}")
                cols[1].code(entry.get("entry_hash", "")[:16] + "...")
                cols[2].caption(entry.get("event", ""))

with tab4:
    st.header("RAG vs GraphRAG Decision Matrix")
    st.caption("Use this table to decide which chat interface (RAG vs GraphRAG) fits your question type.")

    comparison_data = {
        "Dimension": [
            "Query Type",
            "Context Assembly",
            "Best For",
            "Latency",
            "Storage Cost",
            "Setup Complexity",
            "Multi-hop Reasoning",
            "Entity Tracking",
            "Provenance",
        ],
        "Traditional RAG": [
            "Semantic similarity",
            "Top-k chunks",
            "Single-doc Q&A",
            "Fast (~2s)",
            "Low (vectors only)",
            "Simple",
            "❌ Limited",
            "❌ None",
            "Document-level",
        ],
        "GraphRAG": [
            "Entity + Relation",
            "BFS traversal + vectors",
            "Cross-doc analysis",
            "Slower (~5s)",
            "High (graph + vectors)",
            "Complex",
            "✅ 2-3 hop paths",
            "✅ Full lineage",
            "Entity-level",
        ],
    }

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.info("""
    **Decision Rule**:
    - Use **RAG** for: "What does document X say about Y?"
    - Use **GraphRAG** for: "Which vendors are connected to high expenses?" or "Why did Q3 costs spike?"
    """)

st.markdown("---")
st.caption("Architecture implements 19-step deterministic workflow from Dr Data Tier 0 Specification")
