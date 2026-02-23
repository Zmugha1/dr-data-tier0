"""GraphRAG Chat - Knowledge Graph + Vector DB → LLM."""

import json
import streamlit as st

try:
    import ollama
except ImportError:
    ollama = None
try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from core.knowledge_graph import KnowledgeGraphBuilder
from core.vector_store import VectorStore

st.set_page_config(page_title="GraphRAG Chat", layout="wide")

st.title("🕸️ GraphRAG Chat")
st.markdown("""
**Architecture**: Document → Knowledge Graph (Entities/Relations) + Vector DB → LLM  
**Use case**: Multi-hop reasoning, relationship analysis, "Why" questions, entity-centric queries
""")

with st.expander("🏗️ View Architecture"):
    st.graphviz_chart("""
    digraph G {
        rankdir=TB;
        node [shape=box, style=rounded];

        Doc [label="Document", fillcolor=lightblue];
        Extract [label="Entity Extraction\\n(Phi-4)", fillcolor=lightyellow];
        Resolve [label="Entity Resolution\\n(Canonicalization)", fillcolor=lightyellow];
        GraphDB [label="Knowledge Graph\\n(NetworkX)", shape=ellipse, fillcolor=lightgrey];
        VectorDB [label="Vector DB", shape=cylinder, fillcolor=lightgrey];

        Query [label="User Query", fillcolor=pink];
        Parse [label="Parse Entities", fillcolor=lightyellow];
        Traverse [label="Graph Traversal\\n(BFS 2-hop)", fillcolor=orange];
        Retrieve [label="Vector + Graph\\nContext Merge", fillcolor=lightyellow];
        LLM [label="LLM Reasoning\\n(Qwen 14B)", fillcolor=lightcoral];
        Answer [label="Structured Answer\\n(with provenance)", fillcolor=lightgreen];

        Doc -> Extract -> Resolve -> GraphDB;
        Doc -> VectorDB;

        Query -> Parse -> Traverse -> GraphDB -> Retrieve;
        Query -> VectorDB -> Retrieve -> LLM -> Answer;
    }
    """)

if "graph_messages" not in st.session_state:
    st.session_state.graph_messages = []

try:
    kg = KnowledgeGraphBuilder()
except Exception:
    kg = None

with st.sidebar:
    st.header("GraphRAG Configuration")
    model = st.selectbox("Reasoning Model", ["qwen2.5:14b", "llama3.1:8b", "phi4:latest"])
    traversal_depth = st.slider("Graph traversal depth", 1, 3, 2)
    show_graph = st.checkbox("Visualize subgraph", value=True)

st.sidebar.subheader("Entity Explorer")
if kg and kg.G.number_of_nodes() > 0:
    entities = [
        kg.G.nodes[n].get("name", n)
        for n in kg.G.nodes()
        if kg.G.nodes[n].get("type") != "Document"
    ]
    options = ["-- Select --"] + (entities[:20] if entities else [])
    selected_entity = st.sidebar.selectbox("Inspect entity", options)
else:
    selected_entity = None
    st.sidebar.info("Upload documents to see entities")

if selected_entity and selected_entity != "-- Select --" and kg:
    subgraph = kg.query_graph(selected_entity, depth=traversal_depth)
    st.sidebar.json(subgraph.get("nodes", [])[:3], expanded=False)

for msg in st.session_state.graph_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "graph_evidence" in msg:
            with st.expander("Graph Evidence"):
                st.json(msg["graph_evidence"])

query = st.chat_input("Ask about relationships, causes, or entities...")

if query:
    st.session_state.graph_messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Traversing knowledge graph..."):
                vector_store = VectorStore()
                vec_results = vector_store.query(query, n_results=3)

                query_entities = []
                if kg and kg.G.number_of_nodes() > 0:
                    query_words = set(query.lower().split())
                    for n in kg.G.nodes():
                        name = str(kg.G.nodes[n].get("name", "")).lower()
                        if any(w in name for w in query_words):
                            query_entities.append(n)
                    query_entities = query_entities[:2]

                graph_context = []
                if kg:
                    for ent in query_entities:
                        sub = kg.query_graph(
                            kg.G.nodes[ent].get("name", ent), depth=traversal_depth
                        )
                        graph_context.append(sub)

                documents = vec_results.get("documents", [[]])
                vector_context = (
                    "\n".join(documents[0]) if documents else "No vector context."
                )
                graph_context_str = (
                    json.dumps(graph_context, indent=2)
                    if graph_context
                    else "No graph connections found."
                )

                metadatas = vec_results.get("metadatas", [[]])
                meta_list = metadatas[0] if metadatas else []

            if ollama:
                prompt = f"""Answer using BOTH vector similarity results and knowledge graph relationships.
Provide provenance: cite document sources and entity relationships.

Vector Context (Semantic Similarity):
{vector_context}

Graph Context (Entity Relationships):
{graph_context_str}

Question: {query}

Provide your answer in this format:
**Answer**: [Your answer]
**Evidence**: [Cite specific documents and graph relationships]
**Confidence**: [High/Medium/Low based on evidence convergence]"""

                response = ollama.generate(
                    model=model,
                    prompt=prompt,
                    options={"temperature": 0.1},
                )
                answer = response.get("response", "No response.")
            else:
                answer = "Ollama not available."
                st.info("Install: pip install ollama")

            st.write(answer)

            if show_graph and graph_context and go:
                nodes = graph_context[0].get("nodes", [])[:10]
                if nodes:
                    x_pos = list(range(len(nodes)))
                    y_pos = [0] * len(nodes)
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=x_pos,
                            y=y_pos,
                            mode="markers+text",
                            text=[n.get("name", n.get("id", "")) for n in nodes],
                            textposition="top center",
                            marker=dict(size=20, color="lightblue"),
                        )
                    )
                    fig.update_layout(
                        title="Entity Subgraph (Simplified)",
                        showlegend=False,
                        height=200,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.session_state.graph_messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "graph_evidence": {
                        "entities_found": query_entities,
                        "traversal_depth": traversal_depth,
                        "vector_sources": meta_list,
                    },
                }
            )
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Upload documents on the main page and ensure Ollama is running.")
