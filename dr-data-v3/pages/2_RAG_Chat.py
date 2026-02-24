"""RAG Chat + GraphRAG - Simple RAG and multi-hop reasoning"""
import streamlit as st
import plotly.graph_objects as go
import networkx as nx

from utils.llm_manager import LLMManager
from utils.simple_rag import SimpleRAG
from utils.graph_builder import GraphBuilder
from utils.nav import render_nav

st.set_page_config(page_title="RAG Chat", layout="wide")
render_nav()
mgr = LLMManager()
rag = SimpleRAG()

# Check Ollama first
if not mgr.is_ollama_running():
    st.error("## Ollama Not Running")
    st.markdown("""
    **To fix:**
    1. Open Command Prompt or PowerShell
    2. Run: `ollama serve`
    3. Keep that window open
    4. Refresh this page
    """)
    st.code("ollama serve", language="bash")
    with st.expander("💡 Low VRAM / GPU crashing? Use CPU mode"):
        st.markdown("Force CPU-only (slower but stable on laptops):")
        st.code("$env:OLLAMA_NO_GPU=1\nollama serve", language="powershell")
    st.stop()

# Verify models (need nomic + at least one chat model)
if not mgr.ensure_model("nomic-embed-text:latest"):
    st.error("❌ nomic-embed-text not installed. Return to setup.")
    st.stop()
if not mgr.ensure_model("phi4:latest") and not mgr.ensure_model("llama3.1:8b"):
    st.error("❌ Install phi4:latest or llama3.1:8b. Return to setup.")
    st.stop()

# Tabs: Simple RAG | GraphRAG
tab_rag, tab_graph = st.tabs(["💬 Simple RAG", "🕸️ GraphRAG"])

# ============ SIMPLE RAG TAB ============
with tab_rag:
    st.title("💬 RAG Chat")
    stats = rag.get_stats()
    if stats["documents"] == 0:
        st.info("No documents uploaded. Go to Upload Docs first.")
    else:
        model = st.sidebar.selectbox(
            "Model (RAG):",
            ["phi4:latest", "llama3.1:8b", "qwen2.5:7b"],
            help="phi4=fastest/laptop-friendly, llama8b=balanced, qwen7b=reasoning"
        )
        top_k = st.sidebar.slider("Context chunks:", 1, 10, 3)

        if 'rag_messages' not in st.session_state:
            st.session_state.rag_messages = []

        for msg in st.session_state.rag_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("Sources"):
                        for r in msg["sources"]:
                            st.caption(f"{r['doc_name']} (relevance: {r['score']:.2f})")

        query = st.chat_input("Ask about your documents...")

        if query:
            st.session_state.rag_messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    q_embed = mgr.embed(query)
                    results = rag.query(q_embed, top_k=top_k)
                    if not results or results[0]["score"] < 0.3:
                        response_text = "I couldn't find relevant information in your documents."
                        sources = []
                    else:
                        context = "\n\n".join([f"[{r['doc_name']}]: {r['text'][:400]}" for r in results])
                        prompt = f"""Answer based on context. If unsure, say "I don't have enough information."
Context:
{context}
Question: {query}
Answer:"""
                        response_text = mgr.generate(prompt, model=model)
                        sources = results
                    st.write(response_text)
                    if sources:
                        with st.expander("Sources"):
                            for r in sources:
                                st.caption(f"{r['doc_name']} (relevance: {r['score']:.2f})")
                    st.session_state.rag_messages.append({"role": "assistant", "content": response_text, "sources": sources})

        if st.sidebar.button("Clear Chat History"):
            st.session_state.rag_messages = []
            st.rerun()

# ============ GRAPHRAG TAB ============
with tab_graph:
    st.title("🕸️ GraphRAG - Multi-Hop Reasoning")
    st.markdown("Build a knowledge graph from your documents and ask complex questions.")

    if 'graph_builder' not in st.session_state:
        st.session_state.graph_builder = GraphBuilder()
    graph = st.session_state.graph_builder

    with st.sidebar:
        st.header("Graph Controls")
        if st.button("🔨 Build Graph from Documents"):
            docs = rag.list_documents()
            if not docs:
                st.error("No documents! Upload in Document Upload first.")
            else:
                with st.spinner("Building graph..."):
                    for doc in docs:
                        doc_data = rag.documents.get(doc['id'], {})
                        chunks = doc_data.get('chunks', [])
                        if chunks:
                            graph.add_document_to_graph(doc['name'], chunks)
                    st.success("Graph built!")
                    st.rerun()
        gstats = graph.get_stats()
        st.metric("Nodes", gstats["nodes"])
        st.metric("Edges", gstats["edges"])
        if st.button("🗑️ Clear Graph"):
            graph.clear()
            st.rerun()

    gtab1, gtab2 = st.tabs(["💬 Query", "🔍 Visualization"])

    with gtab1:
        query = st.text_input("Your question:", placeholder="What relationships exist in this document?")
        if query and gstats["nodes"] > 0:
            with st.spinner("Thinking..."):
                query_entities = graph.extract_query_entities(query)
                graph_result = graph.query_graph(query_entities, depth=2)
                q_embed = mgr.embed(query)
                semantic_results = rag.query(q_embed, top_k=3)
                context = "\n".join(graph_result["relationships"][:5])
                context += "\n\n" + "\n\n".join([r["text"][:300] for r in semantic_results])
                prompt = f"Answer based on this context:\n{context}\n\nQuestion: {query}\nAnswer:"
                graph_model = "phi4:latest" if mgr.ensure_model("phi4:latest") else "llama3.1:8b"
                response = mgr.generate(prompt, model=graph_model, temperature=0.2)
                st.markdown("### Answer")
                st.write(response)
                with st.expander("See reasoning"):
                    st.write("Entities found:", query_entities)
                    st.write("Relationships:", graph_result["relationships"][:5])
        elif query:
            st.warning("Build the graph first using the sidebar button.")

    with gtab2:
        if gstats["nodes"] == 0:
            st.info("Build the graph first using the sidebar button")
        else:
            viz_data = graph.visualize_graph()
            fig = go.Figure()
            G_vis = nx.DiGraph()
            for edge in viz_data["edges"]:
                G_vis.add_edge(edge["source"], edge["target"])
            try:
                pos = nx.spring_layout(G_vis, k=2)
            except Exception:
                pos = {node: (0, 0) for node in G_vis.nodes()}
            for edge in viz_data["edges"]:
                if edge["source"] in pos and edge["target"] in pos:
                    x0, y0 = pos[edge["source"]]
                    x1, y1 = pos[edge["target"]]
                    fig.add_trace(go.Scatter(
                        x=[x0, x1, None], y=[y0, y1, None],
                        mode='lines', line=dict(width=1, color='#888'),
                        hoverinfo='none', showlegend=False
                    ))
            node_x, node_y, node_text = [], [], []
            for node in viz_data["nodes"]:
                if node["id"] in pos:
                    node_x.append(pos[node["id"]][0])
                    node_y.append(pos[node["id"]][1])
                    node_text.append(node["label"])
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y, mode='markers+text',
                marker=dict(size=20, color='lightblue'),
                text=node_text, textposition="top center",
                hoverinfo='text', showlegend=False
            ))
            fig.update_layout(
                showlegend=False, hovermode='closest',
                margin=dict(b=20, l=20, r=20, t=20),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
