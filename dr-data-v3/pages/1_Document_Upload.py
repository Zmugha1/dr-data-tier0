"""Document upload page with PDF support"""
import io
import streamlit as st
import hashlib

from utils.llm_manager import LLMManager
from utils.simple_rag import SimpleRAG
from utils.nav import render_nav

# Check dependencies
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes"""
    import pdfplumber
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def _process_and_store(name: str, text: str, mgr: LLMManager, rag: SimpleRAG):
    """Process text into chunks and store"""
    with st.spinner("Embedding... This takes ~1 min for large docs"):
        words = text.split()
        chunk_size = 512
        overlap = 50

        chunks = []
        progress_bar = st.progress(0)

        for i in range(0, len(words), chunk_size - overlap):
            chunk_text = " ".join(words[i:i + chunk_size])
            embedding = mgr.embed(chunk_text)
            chunks.append({
                "text": chunk_text,
                "embedding": embedding
            })
            progress = min(1.0, (i + chunk_size) / len(words)) if len(words) > 0 else 1.0
            progress_bar.progress(min(1.0, progress))

        doc_id = rag.add_document(name, chunks)
        st.success(f"✅ Stored {len(chunks)} chunks from '{name}'")
        st.balloons()


st.set_page_config(page_title="Document Upload", layout="wide")
render_nav()

mgr = LLMManager()
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

st.title("📤 Document Upload")

if not PDF_SUPPORT:
    st.warning("PDF support disabled. Install pdfplumber: `pip install pdfplumber`")

rag = SimpleRAG()

# Input method
method = st.radio("Input method:", ["Paste Text", "Upload File"])

if method == "Paste Text":
    name = st.text_input("Document name:", "document_1")
    text = st.text_area("Text content:", height=300)

    if st.button("Process") and text:
        _process_and_store(name, text, mgr, rag)

else:  # File upload
    uploaded = st.file_uploader("Upload file:", type=["txt", "pdf"])
    if uploaded:
        doc_name = st.text_input(
            "Document name:",
            value=uploaded.name.replace('.pdf', '').replace('.txt', ''),
            key="upload_doc_name"
        )
        if uploaded.type == "text/plain":
            text = uploaded.read().decode('utf-8')
            if st.button("Process File"):
                _process_and_store(doc_name or uploaded.name, text, mgr, rag)
        elif "pdf" in uploaded.type and PDF_SUPPORT:
            if st.button("Process PDF"):
                with st.spinner("Extracting PDF..."):
                    pdf_bytes = uploaded.read()
                    text = extract_pdf_text(pdf_bytes)
                    if text.strip():
                        _process_and_store(doc_name or uploaded.name, text, mgr, rag)
                    else:
                        st.error("No text found in PDF. Try copying text manually and using Paste Text.")
        elif "pdf" in uploaded.type and not PDF_SUPPORT:
            st.error("PDF processing not available. Install pdfplumber or paste text manually.")

# Show existing docs
st.divider()
st.subheader("📚 Existing Documents")
stats = rag.get_stats()
st.write(f"Documents: {stats['documents']} | Chunks: {stats['chunks']}")

docs = rag.list_documents()
if docs:
    for doc in docs:
        col1, col2, col3 = st.columns([3, 1, 1])
        col1.write(f"**{doc['name']}**")
        col2.write(f"{doc['chunks']} chunks")
        if col3.button("Delete", key=f"del_{doc['id']}"):
            if doc['id'] in rag.documents:
                del rag.documents[doc['id']]
                rag._save_index()
                st.rerun()

if st.button("🗑️ Clear All Documents"):
    rag.clear_all()
    st.success("Cleared!")
    st.rerun()
