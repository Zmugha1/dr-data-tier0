"""Dr Data V3 - Setup Gated Application"""
import streamlit as st
import subprocess
import sys
import time

# Must be first Streamlit command
st.set_page_config(
    page_title="Dr Data V3 - Setup",
    layout="wide",
    initial_sidebar_state="expanded"
)

from setup_checker import get_setup_checker


def _render_nav_links():
    """Navigation - same as other pages for consistency"""
    from utils.nav import render_nav
    render_nav()


def render_setup_wizard():
    """Render the setup wizard - gates access to main app"""
    with st.sidebar:
        _render_nav_links()
        st.divider()

    st.title("🔧 Dr Data V3 - Setup Wizard")
    st.markdown("**Complete the setup below to access the application**")

    checker = get_setup_checker()

    # Run checks
    with st.spinner("Checking system requirements..."):
        checker.check_ollama()
        checker.check_models()
        checker.check_python_packages()

    # Display status
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. Ollama Server")
        if checker.status["ollama_running"]:
            st.success("✅ Running")
        else:
            st.error("❌ Not Running")
            st.info("""
            **To start Ollama:**
            1. Open Command Prompt/PowerShell
            2. Run: `ollama serve`
            3. Keep that window open
            4. Refresh this page
            """)
            with st.expander("💡 GPU memory issues? Use CPU mode"):
                st.code("$env:OLLAMA_NO_GPU=1\nollama serve", language="powershell")

    with col2:
        st.subheader("2. Required Models")
        required_models = {
            k: v for k, v in checker.REQUIRED_MODELS.items()
            if v["auto_install"]
        }

        for model_id, info in required_models.items():
            installed = checker.status["models"].get(model_id, False)
            if installed:
                st.success(f"✅ {info['name']}")
            else:
                st.error(f"❌ {info['name']}")
                if st.button(f"Install {info['name']}", key=f"install_{model_id}"):
                    with st.spinner(f"Downloading {info['size']}..."):
                        success, msg = checker.install_model(model_id)
                        if success:
                            st.success(msg)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Failed: {msg}")
                            st.code(f"Manual command: ollama pull {model_id}")

    with col3:
        st.subheader("3. Optional Models")
        optional = {
            k: v for k, v in checker.REQUIRED_MODELS.items()
            if not v["auto_install"]
        }

        for model_id, info in optional.items():
            installed = checker.status["models"].get(model_id, False)
            if installed:
                st.success(f"✅ {info['name']}")
            else:
                st.warning(f"⚠️ {info['name']} (Optional)")
                if st.button(f"Install {info['name']}", key=f"opt_{model_id}"):
                    with st.spinner(f"Downloading {info['size']}..."):
                        success, msg = checker.install_model(model_id)
                        if success:
                            st.success(msg)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.info("Large model - try in terminal:")
                            st.code(f"ollama pull {model_id}")

    # Python packages check
    st.divider()
    st.subheader("4. Python Dependencies")

    missing_pkgs = [
        (pkg, checker.PACKAGE_DESCRIPTIONS.get(pkg, pkg))
        for pkg, _ in checker.PYTHON_PACKAGES
        if not checker.status["packages"].get(pkg, False)
    ]

    if missing_pkgs:
        st.warning("Some optional features disabled (e.g. PDF processing)")
        pkg_names = [p[0] for p in missing_pkgs]
        st.code(f"pip install {' '.join(pkg_names)}")

        if st.button("Attempt Auto-Install", key="auto_install_pkgs"):
            with st.spinner("Installing packages..."):
                for pkg, desc in missing_pkgs:
                    try:
                        subprocess.check_call(
                            [sys.executable, "-m", "pip", "install", pkg],
                            capture_output=True,
                            timeout=120
                        )
                        st.success(f"Installed {pkg}")
                    except Exception:
                        st.error(f"Failed to install {pkg}")
                time.sleep(1)
                st.rerun()
    else:
        st.success("✅ All Python packages installed")

    # Final gate
    st.divider()

    if checker.is_ready():
        st.success("## 🎉 Setup Complete! You may now use the app.")
        st.balloons()

        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button("Enter Application", type="primary", use_container_width=True):
                st.session_state.setup_complete = True
                st.rerun()
        with nav_col2:
            if st.button("📤 Go to Upload Docs", use_container_width=True):
                st.switch_page("pages/1_Document_Upload.py")
    else:
        st.error("## ⛔ Setup Incomplete")
        st.warning("Please complete steps 1 and 2 above to continue.")

        with st.expander("Manual Setup Commands"):
            st.code(checker.get_setup_commands(), language="bash")

        # Auto-refresh option
        if st.checkbox("Auto-refresh every 10 seconds (while fixing)", key="auto_refresh"):
            time.sleep(10)
            st.rerun()


def render_main_app():
    """Render the main application (only after setup)"""
    st.sidebar.success("✅ System Ready")

    st.sidebar.title("Navigation")
    _render_nav_links()

    st.title("🏥 Dr Data V3 - Zero-Cloud AI")
    st.markdown("""
    ## 👋 Welcome

    Your air-gapped RAG system is ready:

    - ✅ Ollama server connected
    - ✅ All models installed
    - ✅ PDF processing enabled (if dependencies installed)
    - ✅ GraphRAG for multi-hop reasoning (entity extraction + knowledge graph)

    **Get started:** Use the sidebar. GraphRAG is inside RAG Chat (🕸️ tab).
    """)

    # Quick stats
    from utils.llm_manager import LLMManager
    from utils.simple_rag import SimpleRAG

    mgr = LLMManager()
    rag = SimpleRAG()
    stats = rag.get_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("Models Ready", len([m for m in mgr.list_models() if m['available']]))
    col2.metric("Documents", stats["documents"])
    col3.metric("Status", "Online")

    if st.sidebar.button("🔙 Return to Setup"):
        st.session_state.setup_complete = False
        st.rerun()


# Main gate
if 'setup_complete' not in st.session_state:
    st.session_state.setup_complete = False

if not st.session_state.setup_complete:
    render_setup_wizard()
else:
    render_main_app()
