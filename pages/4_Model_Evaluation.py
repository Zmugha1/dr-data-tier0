"""
DR DATA - TIER 0 LLM PRESCRIPTION ENGINE (REVISED)
==================================================
Now includes Cloud vs Local comparison and Air-Gapped Translation Guide
"""

import json
from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Model Evaluation & Architecture Guide", layout="wide")

# ============================================================================
# CONFIGURATION: Model Specifications with Cloud Equivalents
# ============================================================================

MODEL_SPECS = {
    "phi4": {
        "name": "Phi-4 (3.8B)",
        "vram_gb": 3,
        "latency_toks_per_sec": 45,
        "best_for": ["Data Extraction", "Entity Recognition", "Quick Schema Detection"],
        "architecture": "Traditional RAG",
        "quantization": "Q4_K_M",
        "accuracy_tier": "High (85-90%)",
        "context_window": 16384,
        "strengths": "Speed, low VRAM, structured JSON output",
        "weaknesses": "Limited reasoning depth, math errors",
        "hardware_tier": "Edge/eGPU",
        "cloud_equivalent": "GPT-3.5 Turbo / Claude Haiku (but 100% private, 100x cheaper)",
        "use_case_example": "Processing 10K invoices overnight on a laptop",
    },
    "llama3.1-8b": {
        "name": "Llama 3.1 8B Instruct",
        "vram_gb": 6,
        "latency_toks_per_sec": 22,
        "best_for": ["Contract Analysis", "General Business Q&A", "Tool Use", "Summarization"],
        "architecture": "Both RAG & GraphRAG",
        "quantization": "Q4_K_M",
        "accuracy_tier": "Very High (88-92%)",
        "context_window": 131072,
        "strengths": "Long context (128K), balanced performance, safe/aligned",
        "weaknesses": "Slower than SLMs for simple tasks",
        "hardware_tier": "eGPU/Desktop",
        "cloud_equivalent": "Claude Instant / GPT-3.5 (with longer context)",
        "use_case_example": "Legal contract review with 100-page context window",
    },
    "qwen14b": {
        "name": "Qwen 2.5 14B",
        "vram_gb": 10,
        "latency_toks_per_sec": 15,
        "best_for": ["Financial Analysis", "Complex Reasoning", "Multi-hop Queries", "Code Generation"],
        "architecture": "GraphRAG (Primary)",
        "quantization": "Q4_0",
        "accuracy_tier": "Expert (90-94%)",
        "context_window": 32768,
        "strengths": "Math, logic, multilingual, causal reasoning",
        "weaknesses": "High VRAM, slower for simple extraction",
        "hardware_tier": "eGPU (24GB) / Desktop",
        "cloud_equivalent": "GPT-4 (approximate reasoning capability)",
        "use_case_example": "Multi-document financial audit with causal analysis",
    },
    "deepseek-coder-7b": {
        "name": "DeepSeek-Coder 7B",
        "vram_gb": 5,
        "latency_toks_per_sec": 30,
        "best_for": ["SQL Generation", "Python Automation", "API Integration"],
        "architecture": "Traditional RAG",
        "quantization": "Q4_K_M",
        "accuracy_tier": "High (Code: 80% pass@1)",
        "context_window": 16384,
        "strengths": "Code-first, fill-in-middle, technical docs",
        "weaknesses": "General reasoning weaker than Llama 8B",
        "hardware_tier": "eGPU",
        "cloud_equivalent": "GitHub Copilot / CodeWhisperer (local version)",
        "use_case_example": "Generating SQL from natural language business questions",
    },
    "mixtral-8x7b": {
        "name": "Mixtral 8x7b (MoE)",
        "vram_gb": 28,
        "latency_toks_per_sec": 8,
        "best_for": ["Deep Research", "Multi-Document Audit", "Complex Agentic Planning"],
        "architecture": "GraphRAG (Complex)",
        "quantization": "Q4_K_M",
        "accuracy_tier": "Expert (91-95%)",
        "context_window": 32768,
        "strengths": "Sparse MoE efficiency, highest quality at 47B scale",
        "weaknesses": "Requires dual GPU or CPU offload, slow",
        "hardware_tier": "Desktop (Dual GPU)",
        "cloud_equivalent": "GPT-4 Turbo / Claude 3 Opus (sparse architecture match)",
        "use_case_example": "Enterprise compliance audit across 1000+ documents",
    },
    "llama3.1-70b": {
        "name": "Llama 3.1 70B (Q4)",
        "vram_gb": 40,
        "latency_toks_per_sec": 5,
        "best_for": ["Enterprise Compliance", "High-Stakes Legal", "Medical Analysis"],
        "architecture": "GraphRAG (Critical)",
        "quantization": "Q4_0",
        "accuracy_tier": "State-of-Art (94-96%)",
        "context_window": 131072,
        "strengths": "Near-GPT-4 reasoning, safety, consistency",
        "weaknesses": "Very slow, requires 48GB VRAM or aggressive quantization",
        "hardware_tier": "Workstation (A6000/ Dual 4090)",
        "cloud_equivalent": "GPT-4 / Claude 3.5 Sonnet (frontier model)",
        "use_case_example": "HIPAA-compliant medical record analysis with zero cloud exposure",
    },
}

# ============================================================================
# AIR-GAPPED TRANSLATION GUIDE DATA
# ============================================================================

CLOUD_LOCAL_MAPPING = [
    {
        "category": "LLM APIs",
        "cloud_tool": "OpenAI GPT-4, Claude, Gemini",
        "local_tier0": "Ollama (Phi-4, Llama 3.1, Qwen)",
        "status": "✅ Implemented",
        "notes": "Zero API keys, zero telemetry, 100% offline",
    },
    {
        "category": "Vector Database",
        "cloud_tool": "Pinecone, Weaviate Cloud, Milvus",
        "local_tier0": "ChromaDB (persistent) or LanceDB",
        "status": "✅ Implemented",
        "notes": "File-based storage in ./data/vector_db/",
    },
    {
        "category": "Data Labeling",
        "cloud_tool": "Labelbox, SuperAnnotate, Scale",
        "local_tier0": "Phi-4 auto-labeling + deterministic regex",
        "status": "✅ Implemented",
        "notes": "Automated entity extraction without human labelers",
    },
    {
        "category": "RAG Framework",
        "cloud_tool": "LangChain Cloud, LlamaIndex Cloud",
        "local_tier0": "LangChain Community (offline) + custom orchestration",
        "status": "✅ Implemented",
        "notes": "No callbacks to LangSmith or external telemetry",
    },
    {
        "category": "AI Agents",
        "cloud_tool": "Autogen (Azure OpenAI), MetaGPT",
        "local_tier0": "LangGraph (local) + ReAct loops in Streamlit",
        "status": "⚠️ Tier 1",
        "notes": "Planned for agentic workflow tier",
    },
    {
        "category": "Workflow Automation",
        "cloud_tool": "Zapier, Make, Workato",
        "local_tier0": "n8n (self-hosted) or APScheduler (Python)",
        "status": "⚠️ Optional",
        "notes": "Only needed for scheduled reporting (cron replacement)",
    },
    {
        "category": "Output Validation",
        "cloud_tool": "Guardrails AI (cloud), PromptLayer",
        "local_tier0": "Pydantic + Guardrails (local install) + JSON schemas",
        "status": "✅ Implemented",
        "notes": "Deterministic output validation with constrained decoding",
    },
    {
        "category": "Embeddings",
        "cloud_tool": "OpenAI Ada-002, Cohere Embed",
        "local_tier0": "Nomic-embed-text-v1.5, all-MiniLM-L6-v2",
        "status": "✅ Implemented",
        "notes": "Runs on CPU, no cloud dependency",
    },
    {
        "category": "Graph Database",
        "cloud_tool": "Neo4j Aura (cloud), Amazon Neptune",
        "local_tier0": "NetworkX (in-memory) or Neo4j Community (local)",
        "status": "✅ Implemented",
        "notes": "Serialized to pickle/Parquet for air-gapped portability",
    },
    {
        "category": "Observability",
        "cloud_tool": "Helicone, LangSmith, Weights & Biases",
        "local_tier0": "SQLite audit logs + hash chain (WORM storage)",
        "status": "✅ Implemented",
        "notes": "Complete traceability without external telemetry",
    },
]

# ============================================================================
# DECISION RUBRIC ENGINE
# ============================================================================


@dataclass
class UseCaseProfile:
    task_type: Literal["Extraction", "Analysis", "Generation", "Code", "Compliance"]
    data_volume: Literal["Low (<1K docs)", "Medium (1K-10K)", "High (>10K)"]
    reasoning_depth: Literal["Pattern Match", "Single-hop", "Multi-hop", "Causal/Agentic"]
    latency_requirement: Literal["Real-time (<2s)", "Fast (<5s)", "Batch (OK)"]
    accuracy_criticality: Literal["Draft OK", "Business Decision", "Compliance/Legal"]
    vram_available: int
    multi_document: bool
    relationship_complexity: Literal["None", "Simple Links", "Complex Graph"]


class LLMPrescriptionRubric:
    def __init__(self):
        self.decision_log = []

    def evaluate(self, profile: UseCaseProfile) -> dict:
        architecture = "Traditional RAG"
        confidence = "High"
        warnings = []

        available_models = {
            k: v for k, v in MODEL_SPECS.items() if v["vram_gb"] <= profile.vram_available * 0.8
        }

        if not available_models:
            return {
                "error": "Insufficient VRAM",
                "message": f"Available: {profile.vram_available}GB. Minimum required: 3GB",
                "recommendation": "Upgrade to RTX 4090 (24GB) or reduce quantization",
            }

        if profile.reasoning_depth in ["Multi-hop", "Causal/Agentic"] or profile.multi_document:
            architecture = "GraphRAG"
            reasoning_models = ["qwen14b", "llama3.1-8b", "mixtral-8x7b", "llama3.1-70b"]
            available_models = {k: v for k, v in available_models.items() if k in reasoning_models}
            if not available_models:
                warnings.append(
                    "GraphRAG requires 8B+ models. Falling back to Traditional RAG with preprocessing."
                )
                architecture = "Traditional RAG (Enhanced)"

        if profile.latency_requirement == "Real-time (<2s)":
            fast_models = ["phi4", "deepseek-coder-7b"]
            available_models = {k: v for k, v in available_models.items() if k in fast_models}
            if not available_models:
                warnings.append("Real-time not possible with current VRAM. Consider Phi-4 (3GB).")

        if profile.accuracy_criticality == "Compliance/Legal":
            safe_models = ["llama3.1-8b", "llama3.1-70b", "qwen14b"]
            available_models = {k: v for k, v in available_models.items() if k in safe_models}
            confidence = "Medium" if "llama3.1-70b" not in available_models else "High"

        scores = {}
        for model_id, specs in available_models.items():
            score = 0
            score += (profile.vram_available - specs["vram_gb"]) * 2

            if profile.latency_requirement == "Real-time (<2s)":
                score += specs["latency_toks_per_sec"] * 0.5
            elif profile.latency_requirement == "Batch (OK)":
                score -= specs["latency_toks_per_sec"] * 0.2

            if profile.accuracy_criticality == "Compliance/Legal":
                if "Expert" in specs["accuracy_tier"] or "State" in specs["accuracy_tier"]:
                    score += 50
            elif profile.accuracy_criticality == "Business Decision":
                if "Very High" in specs["accuracy_tier"] or "Expert" in specs["accuracy_tier"]:
                    score += 30

            if profile.task_type == "Extraction" and model_id == "phi4":
                score += 40
            if profile.task_type == "Code" and model_id == "deepseek-coder-7b":
                score += 40
            if profile.task_type in ["Analysis", "Compliance"] and model_id in [
                "qwen14b",
                "llama3.1-70b",
            ]:
                score += 40

            if profile.multi_document and specs["vram_gb"] < 6:
                score -= 100

            scores[model_id] = score

        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_model = sorted_models[0][0] if sorted_models else "llama3.1-8b"
        secondary_model = sorted_models[1][0] if len(sorted_models) > 1 else None

        prescription = {
            "primary_model": primary_model,
            "primary_details": MODEL_SPECS[primary_model],
            "secondary_model": secondary_model,
            "secondary_details": MODEL_SPECS[secondary_model] if secondary_model else None,
            "recommended_architecture": architecture,
            "confidence": confidence,
            "warnings": warnings,
            "use_case_profile": profile,
            "justification": self._generate_justification(
                profile, primary_model, architecture, warnings
            ),
            "implementation_notes": self._get_implementation_notes(primary_model, architecture),
        }

        return prescription

    def _generate_justification(self, profile, model_id, architecture, warnings):
        model = MODEL_SPECS[model_id]
        reasons = [
            f"Selected **{model['name']}** based on:",
            f"1. **Hardware Fit**: Requires {model['vram_gb']}GB VRAM (Available: {profile.vram_available}GB)",
            f"2. **Speed**: {model['latency_toks_per_sec']} tok/s matches '{profile.latency_requirement}' requirement",
            f"3. **Accuracy**: {model['accuracy_tier']} tier suitable for '{profile.accuracy_criticality}' use case",
            f"4. **Architecture**: {architecture} recommended for '{profile.reasoning_depth}' reasoning depth",
            f"5. **Cloud Equivalent**: Comparable to {model['cloud_equivalent']}",
        ]

        if profile.multi_document:
            reasons.append(
                "6. **Multi-document**: Enabled graph-based retrieval for cross-document relationships"
            )

        if warnings:
            reasons.append(f"\n⚠️ **Considerations**: {'; '.join(warnings)}")

        return "\n".join(reasons)

    def _get_implementation_notes(self, model_id, architecture):
        notes = []
        model = MODEL_SPECS[model_id]

        notes.append(f"**Model Download**: `ollama pull {model_id}`")
        notes.append(f"**Quantization**: Use {model['quantization']} for optimal VRAM usage")

        if architecture == "GraphRAG":
            notes.append(
                "**Graph Setup**: Install Neo4j locally: `docker run -p 7474:7474 -p 7687:7687 neo4j`"
            )
            notes.append(
                "**Pipeline**: Document → Phi-4 (Entity Extraction) → Neo4j → Qwen (Reasoning)"
            )
        else:
            notes.append("**Pipeline**: Document → ChromaDB (Vectors) → Direct LLM Query")

        if model_id in ["qwen14b", "mixtral-8x7b"]:
            notes.append("**Thermal**: Monitor eGPU temps - sustained inference may throttle")

        return notes


# ============================================================================
# STREAMLIT UI WITH DROPDOWN TABS
# ============================================================================


def render_model_evaluation_tab():
    st.title("🧠 Dr Data: LLM Prescription Engine")
    st.markdown("""
    **What this page does**: Get a tailored recommendation for which local LLM to use based on your use case, hardware, and quality needs.  
    No models are run here—this is a diagnostic tool that maps your requirements to the best offline stack (Phi-4, Llama, Qwen, etc.).

    **How to use**:  
    1. **Explore** — Open the expanders above to read about Tier 0 value, cloud-to-local mappings, and RAG vs GraphRAG.  
    2. **Configure** — Fill in the form below: task type, document volume, reasoning depth, latency, accuracy, and VRAM.  
    3. **Prescribe** — Click **Generate Prescription** to receive a model recommendation with justification and implementation commands.  
    4. **Export** — Download the prescription as JSON for deployment planning.
    """)

    # DROPDOWN 1: Why Tier 0?
    with st.expander("🏔️ Why Tier 0? Understanding Air-Gapped AI Architecture", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("The Problem with Cloud AI")
            st.markdown("""
            - **Data Exposure**: Every document sent to OpenAI/Claude is stored for training
            - **Compliance Violations**: HIPAA, GDPR, SOX prohibit third-party AI processing
            - **Vendor Lock-in**: $2-5 per 1K tokens adds up to $10K+/month for active use
            - **Internet Dependency**: No analysis possible during outages or in secure facilities
            - **Version Drift**: GPT-4 behavior changes unpredictably between API calls
            """)

        with col2:
            st.subheader("The Tier 0 Solution")
            st.markdown("""
            - **Zero Data Exposure**: Documents never leave your machine (SHA-256 hashed locally)
            - **Compliance Native**: Air-gapped by design—no network = no violation
            - **Fixed Costs**: One-time hardware ($6K) vs perpetual cloud fees ($120K/3 years)
            - **Deterministic**: Same input → Same output, every time (seed-locked, version-pinned)
            - **Sovereign**: You own the model weights, the data, and the inference engine
            """)

        st.info("""
        **Use This Tool When**: A client asks "Can we use AI without sending our data to the cloud?" 
        or when you need to prove compliance for regulated industries (healthcare, legal, finance).
        """)

    # DROPDOWN 2: Cloud vs Local Translation Guide
    with st.expander("🔄 Cloud-to-Local Translation Guide", expanded=False):
        st.markdown("""
        **Mapping Enterprise AI Tools to Air-Gapped Equivalents**  
        *"What do I use instead of [Cloud Tool]?"*
        """)

        df_mapping = pd.DataFrame(CLOUD_LOCAL_MAPPING)

        def color_status(val):
            if val == "✅ Implemented":
                return "background-color: #d4edda; color: #155724"
            if val == "⚠️ Tier 1":
                return "background-color: #fff3cd; color: #856404"
            if val == "⚠️ Optional":
                return "background-color: #cce5ff; color: #004085"
            return ""

        try:
            styled_df = df_mapping.style.map(color_status, subset=["status"])
        except AttributeError:
            styled_df = df_mapping.style.applymap(color_status, subset=["status"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.subheader("What You Can Ignore (Cloud-Only)")
        st.markdown("""
        ❌ **Claude/OpenAI/Gemini APIs**: Explicitly replaced by local Ollama  
        ❌ **Zapier/Make**: Requires webhooks; use Python scheduling instead  
        ❌ **PromptLayer**: Telemetry tool; violates air-gap principles  
        ❌ **Pinecone Cloud**: Proprietary vector DB; use ChromaDB  
        ❌ **AWS SageMaker**: Managed notebooks; use local Jupyter  
        """)

        st.subheader("What You Must Master (Tier 0 Critical)")
        st.markdown("""
        ✅ **Ollama**: Local LLM server (replaces OpenAI client)  
        ✅ **ChromaDB**: File-based vector store (replaces Pinecone)  
        ✅ **NetworkX**: In-memory graph analytics (replaces Neo4j Aura)  
        ✅ **Pydantic/JSON Schema**: Output validation (replaces Guardrails Cloud)  
        ✅ **SHA-256 Hashing**: Content-addressable storage for idempotency  
        """)

    # DROPDOWN 3: Architecture Comparison Deep Dive
    with st.expander("🏗️ Architecture: Traditional RAG vs GraphRAG", expanded=False):
        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.subheader("Traditional RAG")
            st.graphviz_chart("""
            digraph {
                rankdir=TB;
                node [shape=box, style=rounded, fillcolor=lightblue];
                A[label="Document"] -> B[label="Chunk (512 tokens)"] -> C[label="Embedding"];
                C -> D[shape=cylinder, label="Vector DB", fillcolor=lightgrey];
                Q[label="Query", fillcolor=pink] -> D;
                D -> E[label="Top-k Similarity"];
                E -> F[label="LLM (Llama 3.1)"];
                F -> G[label="Answer"];
            }
            """)
            st.markdown("""
            **Best for**: Single-document Q&A, semantic search, summarization  
            **Latency**: Fast (~2s)  
            **Memory**: Low (vectors only)  
            **Limitation**: Cannot answer "Why did expenses spike?" (no causal reasoning)
            """)

        with comp_col2:
            st.subheader("GraphRAG (Multi-Hop)")
            st.graphviz_chart("""
            digraph {
                rankdir=TB;
                node [shape=box, style=rounded, fillcolor=lightblue];
                A[label="Document"] -> B[label="Entity Extraction (Phi-4)"];
                B -> C[label="Entity Resolution"];
                C -> D[shape=ellipse, label="Knowledge Graph", fillcolor=lightyellow];
                Q[label="Query", fillcolor=pink] -> D;
                D -> E[label="BFS Traversal (2-hop)"];
                E -> F[label="Context Assembly"];
                F -> G[label="LLM (Qwen 14B)"];
                G -> H[label="Cited Answer"];
            }
            """)
            st.markdown("""
            **Best for**: Multi-document audits, supply chain analysis, root cause analysis  
            **Latency**: Slower (~5s)  
            **Memory**: High (graph + vectors)  
            **Advantage**: Can trace "Vendor X → Product Y → Expense Z" relationships
            """)

        st.divider()
        st.subheader("Decision Matrix")
        decision_data = {
            "Question Type": [
                "What does the contract say about liability?",
                "Which vendors supply my highest-cost items?",
                "Summarize this 100-page document",
                "Why did Q3 operating expenses increase?",
                "Find all documents mentioning 'force majeure'",
                "Detect conflicts of interest across 50 contracts",
            ],
            "Recommended": [
                "Traditional RAG",
                "GraphRAG",
                "Traditional RAG",
                "GraphRAG",
                "Traditional RAG",
                "GraphRAG",
            ],
            "Reason": [
                "Single-document semantic search",
                "Requires multi-hop vendor→product→cost traversal",
                "Long-context summarization (128K window)",
                "Causal reasoning requires entity relationship graph",
                "Keyword + semantic similarity sufficient",
                "Cross-document entity resolution + path finding",
            ],
        }
        st.dataframe(pd.DataFrame(decision_data), use_container_width=True, hide_index=True)

    # MAIN PRESCRIPTION INTERFACE
    st.divider()
    st.header("📋 Prescription Configuration")
    st.caption("Answer the questions below, then click **Generate Prescription** to get your recommendation.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Use Case Characteristics")

        task_type = st.selectbox(
            "Primary Task Type",
            options=["Extraction", "Analysis", "Generation", "Code", "Compliance"],
            help="What is the main AI task? Extraction = pulling data from docs, Analysis = insights, etc.",
        )

        data_volume = st.select_slider(
            "Document Volume",
            options=["Low (<1K docs)", "Medium (1K-10K)", "High (>10K)"],
            value="Medium (1K-10K)",
        )

        reasoning_depth = st.selectbox(
            "Reasoning Complexity",
            options=["Pattern Match", "Single-hop", "Multi-hop", "Causal/Agentic"],
            help="Pattern Match = regex-like, Single-hop = one document, Multi-hop = connect docs, Causal = why questions",
        )

        multi_doc = st.checkbox(
            "Multi-Document Analysis Required",
            help="Check if you need to connect information across multiple files (e.g., audit trails)",
        )

        relationship_complexity = st.select_slider(
            "Relationship Complexity",
            options=["None", "Simple Links", "Complex Graph"],
            value="Simple Links" if multi_doc else "None",
        )

    with col2:
        st.subheader("⚙️ Hardware & Quality Constraints")

        latency_req = st.radio(
            "Latency Requirement",
            options=["Real-time (<2s)", "Fast (<5s)", "Batch (OK)"],
            index=1,
        )

        accuracy_req = st.radio(
            "Accuracy Criticality",
            options=["Draft OK", "Business Decision", "Compliance/Legal"],
            index=1,
            help="Compliance = Medical/Legal/Financial high stakes",
        )

        vram_available = st.number_input(
            "Available VRAM (GB)",
            min_value=4,
            max_value=80,
            value=24,
            help="Check nvidia-smi. Leave 20% buffer for OS/other apps",
        )

        hardware_type = st.radio(
            "Hardware Setup",
            options=["Laptop + eGPU (TB4)", "Desktop Workstation", "Edge Device (No GPU)"],
        )

    st.divider()
    st.caption("Click the button below after selecting your options above.")
    if st.button("🔍 Generate Prescription", type="primary", use_container_width=True):
        profile = UseCaseProfile(
            task_type=task_type,
            data_volume=data_volume,
            reasoning_depth=reasoning_depth,
            latency_requirement=latency_req,
            accuracy_criticality=accuracy_req,
            vram_available=vram_available,
            multi_document=multi_doc,
            relationship_complexity=relationship_complexity,
        )

        rubric = LLMPrescriptionRubric()
        prescription = rubric.evaluate(profile)

        if "error" in prescription:
            st.error(prescription["error"])
            st.info(prescription["recommendation"])
        else:
            st.success("## 💊 Prescription Generated")

            res_col1, res_col2 = st.columns([2, 1])

            with res_col1:
                st.markdown(f"### Primary: {prescription['primary_details']['name']}")
                st.markdown(f"""
                **Architecture**: {prescription['recommended_architecture']}  
                **Confidence**: {prescription['confidence']}  
                **Speed**: {prescription['primary_details']['latency_toks_per_sec']} tokens/sec  
                **VRAM**: {prescription['primary_details']['vram_gb']} GB required  
                **Cloud Equivalent**: *{prescription['primary_details']['cloud_equivalent']}*
                """)

                if prescription["secondary_model"]:
                    st.markdown(
                        f"### Fallback: {prescription['secondary_details']['name']}"
                    )
                    st.caption("Use if primary fails or for preprocessing steps")

            with res_col2:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=prescription["primary_details"]["vram_gb"],
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "VRAM Usage"},
                        delta={"reference": vram_available, "relative": True},
                        gauge={
                            "axis": {"range": [None, vram_available]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {
                                    "range": [0, vram_available * 0.6],
                                    "color": "lightgray",
                                },
                                {
                                    "range": [
                                        vram_available * 0.6,
                                        vram_available * 0.9,
                                    ],
                                    "color": "yellow",
                                },
                                {
                                    "range": [
                                        vram_available * 0.9,
                                        vram_available,
                                    ],
                                    "color": "red",
                                },
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": vram_available * 0.8,
                            },
                        },
                    )
                )
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📖 Clinical Justification")
            st.markdown(prescription["justification"])

            st.info(
                f"**Example Use Case**: {prescription['primary_details']['use_case_example']}"
            )

            if prescription["warnings"]:
                for warning in prescription["warnings"]:
                    st.warning(warning)

            st.markdown("### 🛠️ Implementation Commands")
            for note in prescription["implementation_notes"]:
                st.code(note, language="bash")

            st.markdown("### 🏗️ Recommended Architecture")
            if prescription["recommended_architecture"] == "GraphRAG":
                st.graphviz_chart("""
                digraph {
                    User -> Streamlit;
                    Streamlit -> ChromaDB [label="Vector Search"];
                    Streamlit -> Neo4j [label="Graph Query"];
                    ChromaDB -> "Phi-4 (Extract)" -> Neo4j;
                    Neo4j -> "Qwen/Llama (Reason)";
                    "Qwen/Llama (Reason)" -> Prescription;
                }
                """)
            else:
                st.graphviz_chart("""
                digraph {
                    User -> Streamlit;
                    Streamlit -> ChromaDB [label="Similarity"];
                    ChromaDB -> "Phi-4/Llama (Generate)";
                    "Phi-4/Llama (Generate)" -> Answer;
                }
                """)

            st.divider()

            def serialize_prescription(p):
                out = {k: v for k, v in p.items() if k != "use_case_profile"}
                out["use_case_profile"] = asdict(p["use_case_profile"])
                return out

            export_data = {
                "prescription": serialize_prescription(prescription),
                "timestamp": pd.Timestamp.now().isoformat(),
                "tier": "Tier 0 (Air-Gapped)",
            }

            if st.download_button(
                label="📥 Export Prescription (JSON)",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"tier0_prescription_{task_type.lower()}_{pd.Timestamp.now().strftime('%Y%m%d')}.json",
                mime="application/json",
            ):
                st.success("Prescription saved! Use this to configure your deployment.")


# Execute when loaded as Streamlit page
render_model_evaluation_tab()
