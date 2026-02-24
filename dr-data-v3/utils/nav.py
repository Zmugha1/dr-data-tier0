"""Shared navigation - appears in sidebar on every page"""
import streamlit as st


def render_nav():
    """Render consistent nav buttons in sidebar - call this at top of each page's sidebar"""
    st.sidebar.markdown("**Navigate**")
    if st.sidebar.button("🏠 Home", key="nav_home", use_container_width=True):
        st.switch_page("app.py")
    if st.sidebar.button("📤 Upload Docs", key="nav_upload", use_container_width=True):
        st.switch_page("pages/1_Document_Upload.py")
    if st.sidebar.button("💬 RAG Chat", key="nav_rag", use_container_width=True):
        st.switch_page("pages/2_RAG_Chat.py")
    if st.sidebar.button("🕸️ GraphRAG", key="nav_graph", use_container_width=True):
        st.switch_page("pages/2_RAG_Chat.py")
    st.sidebar.caption("GraphRAG = tab in RAG Chat")
    st.sidebar.divider()
