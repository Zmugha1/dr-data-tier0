"""Traditional RAG Chat - Document -> Vector DB -> LLM."""

import streamlit as st

try:
    import ollama
except ImportError:
    ollama = None

from core.vector_store import VectorStore

st.set_page_config(page_title="RAG Chat - Traditional", layout="wide")

st.title("Traditional RAG Chat")
st.markdown("""
**Architecture**: Document -> Vector DB (Similarity Search) -> LLM
**Use case**: Single-document Q&A, semantic search, summarization
""")

with st.expander("View Architecture"):
    st.graphviz_chart("""
    digraph G {
        rankdir=TB;
        node [shape=box, style=rounded];
        Doc [label="Document"];
        Chunk [label="Chunking"];
        Embed [label="Embedding"];
        VectorDB [label="Vector DB"];
        Query [label="User Query"];
        Retrieve [label="Similarity Search"];
        LLM [label="LLM"];
        Answer [label="Answer"];
        Doc -> Chunk -> Embed -> VectorDB;
        Query -> Retrieve -> VectorDB -> Retrieve -> LLM -> Answer;
    }
    """)

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

with st.sidebar:
    st.header("RAG Configuration")
    model = st.selectbox("Model", ["llama3.1:8b", "phi4:latest", "qwen2.5:14b"])
    top_k = st.slider("Retrieval chunks (k)", 1, 10, 5)
    show_context = st.checkbox("Show retrieved context", value=True)

for msg in st.session_state.rag_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and show_context:
            with st.expander("Sources"):
                for src in msg["sources"]:
                    dist = src.get("distance", 0)
                    st.caption(f"From: {src['source']} (Score: {dist:.3f})")

query = st.chat_input("Ask about your documents...")

if query:
    st.session_state.rag_messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving context..."):
                vector_store = VectorStore()
                results = vector_store.query(query, n_results=top_k)

                documents = results.get("documents", [[]])
                metadatas = results.get("metadatas", [[]])
                distances = results.get("distances", [[]])

                context_list = documents[0] if documents else []
                meta_list = metadatas[0] if metadatas else []
                dist_list = distances[0] if distances else []

                context = "\n\n".join(context_list)
                sources = [
                    {
                        "source": m.get("source", "unknown") if isinstance(m, dict) else "unknown",
                        "distance": dist_list[i] if i < len(dist_list) else 0,
                        "chunk": i,
                    }
                    for i, m in enumerate(meta_list)
                ]

            if not context.strip():
                st.warning("No documents indexed yet. Upload files on the main page.")
            elif ollama:
                prompt = f"""Answer based on the following context. Cite sources using [Source: name].

Context:
{context}

Question: {query}

Answer:"""
                response = ollama.generate(
                    model=model,
                    prompt=prompt,
                    options={"temperature": 0.1},
                )
                answer = response.get("response", "No response.")
                st.write(answer)
            else:
                answer = "Ollama not available. Install: pip install ollama"
                st.write(answer)

            st.session_state.rag_messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Ensure documents are uploaded and Ollama is running with nomic-embed-text.")
