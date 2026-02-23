"""
DR DATA CORE ARCHITECTURE PLANNING - LLM Prescription Engine (Model Evaluation Tab)
===================================================================================
A diagnostic rubric that evaluates business use cases against
hardware constraints, accuracy requirements, and architectural
complexity to prescribe the optimal offline LLM stack.
"""

import json
import streamlit as st
from dataclasses import asdict, dataclass
from typing import Literal, Optional

import plotly.graph_objects as go

# ============================================================================
# CONFIGURATION: Model Specifications Database (From Previous Tables)
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
        "hardware_tier": "Edge/eGPU"
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
        "strengths": "Long context, balanced performance, safe/aligned",
        "weaknesses": "Slower than SLMs for simple tasks",
        "hardware_tier": "eGPU/Desktop"
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
        "hardware_tier": "eGPU (24GB) / Desktop"
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
        "hardware_tier": "eGPU"
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
        "hardware_tier": "Desktop (Dual GPU)"
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
        "hardware_tier": "Workstation (A6000/ Dual 4090)"
    }
}


@st.cache_data
def get_model_specs():
    """Cached model specs to prevent re-loading on every interaction."""
    return MODEL_SPECS


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
    vram_available: int  # GB
    multi_document: bool
    relationship_complexity: Literal["None", "Simple Links", "Complex Graph"]

class LLMPrescriptionRubric:
    """
    The Dr Data Diagnostic Algorithm
    Evaluates use case against constraints to prescribe model stack
    """

    def __init__(self):
        self.decision_log = []

    def evaluate(self, profile: UseCaseProfile) -> dict:
        """
        Main diagnostic logic implementing the rubric tables
        """
        candidates = []
        architecture = "Traditional RAG"
        confidence = "High"
        warnings = []

        # --- DECISION NODE 1: Hardware Constraint (Hard Filter) ---
        specs = get_model_specs()
        available_models = {
            k: v for k, v in specs.items()
            if v["vram_gb"] <= profile.vram_available * 0.8  # 20% buffer
        }

        if not available_models:
            return {
                "error": "Insufficient VRAM",
                "message": f"Available: {profile.vram_available}GB. Minimum required: 3GB",
                "recommendation": "Upgrade to RTX 4090 (24GB) or reduce quantization"
            }

        # --- DECISION NODE 2: Task Complexity & Architecture ---
        if profile.reasoning_depth in ["Multi-hop", "Causal/Agentic"] or profile.multi_document:
            architecture = "GraphRAG"
            reasoning_models = ["qwen14b", "llama3.1-8b", "mixtral-8x7b", "llama3.1-70b"]
            available_models = {k: v for k, v in available_models.items() if k in reasoning_models}

            if not available_models:
                warnings.append("GraphRAG requires 8B+ models. Falling back to Traditional RAG with preprocessing.")
                architecture = "Traditional RAG (Enhanced)"

        # --- DECISION NODE 3: Latency Requirements ---
        if profile.latency_requirement == "Real-time (<2s)":
            fast_models = ["phi4", "deepseek-coder-7b"]
            available_models = {k: v for k, v in available_models.items() if k in fast_models}
            if not available_models:
                warnings.append("Real-time not possible with current VRAM. Consider Phi-4 (3GB).")

        # --- DECISION NODE 4: Accuracy Criticality ---
        if profile.accuracy_criticality == "Compliance/Legal":
            safe_models = ["llama3.1-8b", "llama3.1-70b", "qwen14b"]
            available_models = {k: v for k, v in available_models.items() if k in safe_models}
            confidence = "Medium" if "llama3.1-70b" not in available_models else "High"

        # --- DECISION NODE 5: Data Volume Optimization ---
        if profile.data_volume == "High (>10K)" and profile.task_type == "Extraction":
            if "phi4" in available_models:
                candidates.append(("phi4", "Primary: Speed for batch processing"))

        # --- SCORING LOGIC ---
        scores = {}
        for model_id, model_spec in available_models.items():
            score = 0

            # VRAM Efficiency (prefer smaller if sufficient)
            score += (profile.vram_available - model_spec["vram_gb"]) * 2

            # Speed match
            if profile.latency_requirement == "Real-time (<2s)":
                score += model_spec["latency_toks_per_sec"] * 0.5
            elif profile.latency_requirement == "Batch (OK)":
                score -= model_spec["latency_toks_per_sec"] * 0.2

            # Accuracy match
            if profile.accuracy_criticality == "Compliance/Legal":
                if "Expert" in model_spec["accuracy_tier"] or "State" in model_spec["accuracy_tier"]:
                    score += 50
            elif profile.accuracy_criticality == "Business Decision":
                if "Very High" in model_spec["accuracy_tier"] or "Expert" in model_spec["accuracy_tier"]:
                    score += 30

            # Task match
            if profile.task_type == "Extraction" and model_id == "phi4":
                score += 40
            if profile.task_type == "Code" and model_id == "deepseek-coder-7b":
                score += 40
            if profile.task_type in ["Analysis", "Compliance"] and model_id in ["qwen14b", "llama3.1-70b"]:
                score += 40

            # Multi-doc penalty (small models fail here)
            if profile.multi_document and model_spec["vram_gb"] < 6:
                score -= 100

            scores[model_id] = score

        # Select top 2
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_model = sorted_models[0][0] if sorted_models else "llama3.1-8b"
        secondary_model = sorted_models[1][0] if len(sorted_models) > 1 else None

        # --- FINAL PRESCRIPTION ---
        prescription = {
            "primary_model": primary_model,
            "primary_details": specs[primary_model],
            "secondary_model": secondary_model,
            "secondary_details": specs[secondary_model] if secondary_model else None,
            "recommended_architecture": architecture,
            "confidence": confidence,
            "warnings": warnings,
            "use_case_profile": profile,
            "justification": self._generate_justification(
                profile, primary_model, architecture, warnings, specs
            ),
            "implementation_notes": self._get_implementation_notes(
                primary_model, architecture, specs
            )
        }

        return prescription

    def _generate_justification(self, profile, model_id, architecture, warnings, specs: dict):
        """Human-readable explanation of the decision"""
        model = specs[model_id]

        reasons = [
            f"Selected **{model['name']}** based on:",
            f"1. **Hardware Fit**: Requires {model['vram_gb']}GB VRAM (Available: {profile.vram_available}GB)",
            f"2. **Speed**: {model['latency_toks_per_sec']} tok/s matches '{profile.latency_requirement}' requirement",
            f"3. **Accuracy**: {model['accuracy_tier']} tier suitable for '{profile.accuracy_criticality}' use case",
            f"4. **Architecture**: {architecture} recommended for '{profile.reasoning_depth}' reasoning depth"
        ]

        if profile.multi_document:
            reasons.append("5. **Multi-document**: Enabled graph-based retrieval for cross-document relationships")

        if warnings:
            reasons.append(f"\n**Considerations**: {'; '.join(warnings)}")

        return "\n".join(reasons)

    def _get_implementation_notes(self, model_id, architecture, specs: dict):
        """Technical setup instructions"""
        notes = []
        model = specs[model_id]

        notes.append(f"**Model Download**: `ollama pull {model_id}`")
        notes.append(f"**Quantization**: Use {model['quantization']} for optimal VRAM usage")

        if architecture == "GraphRAG":
            notes.append("**Graph Setup**: Install Neo4j locally: `docker run -p 7474:7474 -p 7687:7687 neo4j`")
            notes.append("**Pipeline**: Document -> Phi-4 (Entity Extraction) -> Neo4j -> Qwen (Reasoning)")
        else:
            notes.append("**Pipeline**: Document -> ChromaDB (Vectors) -> Direct LLM Query")

        if model_id in ["qwen14b", "mixtral-8x7b"]:
            notes.append("**Thermal**: Monitor eGPU temps - sustained inference may throttle")

        return notes


# ============================================================================
# STREAMLIT UI (The Dashboard Tab)
# ============================================================================

def render_model_evaluation_tab():
    st.title("Dr Data Core Architecture Planning: LLM Prescription Engine")
    st.markdown("""
    **Diagnostic Rubric for Offline LLM Selection**
    Answer the questions below to receive a customized model recommendation
    based on your business use case, hardware constraints, and accuracy requirements.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Use Case Characteristics")

        task_type = st.selectbox(
            "Primary Task Type",
            options=["Extraction", "Analysis", "Generation", "Code", "Compliance"],
            help="What is the main AI task? Extraction = pulling data from docs, Analysis = insights, etc."
        )

        data_volume = st.select_slider(
            "Document Volume",
            options=["Low (<1K docs)", "Medium (1K-10K)", "High (>10K)"],
            value="Medium (1K-10K)"
        )

        reasoning_depth = st.selectbox(
            "Reasoning Complexity",
            options=["Pattern Match", "Single-hop", "Multi-hop", "Causal/Agentic"],
            help="Pattern Match = regex-like, Single-hop = one document, Multi-hop = connect docs, Causal = why questions"
        )

        multi_doc = st.checkbox(
            "Multi-Document Analysis Required",
            help="Check if you need to connect information across multiple files (e.g., audit trails)"
        )

        relationship_complexity = st.select_slider(
            "Relationship Complexity",
            options=["None", "Simple Links", "Complex Graph"],
            value="Simple Links" if multi_doc else "None"
        )

    with col2:
        st.subheader("Constraints & Requirements")

        latency_req = st.radio(
            "Latency Requirement",
            options=["Real-time (<2s)", "Fast (<5s)", "Batch (OK)"],
            index=1
        )

        accuracy_req = st.radio(
            "Accuracy Criticality",
            options=["Draft OK", "Business Decision", "Compliance/Legal"],
            index=1,
            help="Compliance = Medical/Legal/Financial high stakes"
        )

        vram_available = st.number_input(
            "Available VRAM (GB)",
            min_value=4,
            max_value=80,
            value=24,
            help="Check nvidia-smi. Leave 20% buffer for OS/other apps"
        )

        hardware_type = st.radio(
            "Hardware Setup",
            options=["Laptop + eGPU (TB4)", "Desktop Workstation", "Edge Device (No GPU)"]
        )

    # Evaluate Button
    st.divider()
    if st.button("Generate Prescription", type="primary", use_container_width=True):
        profile = UseCaseProfile(
            task_type=task_type,
            data_volume=data_volume,
            reasoning_depth=reasoning_depth,
            latency_requirement=latency_req,
            accuracy_criticality=accuracy_req,
            vram_available=vram_available,
            multi_document=multi_doc,
            relationship_complexity=relationship_complexity
        )

        rubric = LLMPrescriptionRubric()
        prescription = rubric.evaluate(profile)

        # Handle error case
        if "error" in prescription:
            st.error(f"**{prescription['error']}**: {prescription['message']}")
            st.info(prescription["recommendation"])
            return

        # Display Results
        st.success("## Prescription Generated")

        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            st.markdown(f"### Primary: {prescription['primary_details']['name']}")
            st.markdown(f"""
            **Architecture**: {prescription['recommended_architecture']}
            **Confidence**: {prescription['confidence']}
            **Speed**: {prescription['primary_details']['latency_toks_per_sec']} tokens/sec
            **VRAM**: {prescription['primary_details']['vram_gb']} GB required
            """)

            if prescription['secondary_model']:
                st.markdown(f"### Fallback: {prescription['secondary_details']['name']}")
                st.caption("Use if primary fails or for preprocessing steps")

        with res_col2:
            # VRAM Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prescription['primary_details']['vram_gb'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "VRAM Usage"},
                delta={'reference': vram_available, 'relative': True},
                gauge={
                    'axis': {'range': [None, vram_available]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, vram_available*0.6], 'color': "lightgray"},
                        {'range': [vram_available*0.6, vram_available*0.9], 'color': "yellow"},
                        {'range': [vram_available*0.9, vram_available], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': vram_available * 0.8
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

        # Justification
        st.markdown("### Clinical Justification")
        st.markdown(prescription['justification'])

        # Warnings
        if prescription['warnings']:
            for warning in prescription['warnings']:
                st.warning(warning)

        # Implementation
        st.markdown("### Implementation Notes")
        for note in prescription['implementation_notes']:
            st.code(note, language='bash')

        # Architecture Diagram
        st.markdown("### Recommended Architecture")
        if prescription['recommended_architecture'] == "GraphRAG":
            st.graphviz_chart("""
            digraph {
                User -> Streamlit
                Streamlit -> ChromaDB [label="Vector Search"]
                Streamlit -> Neo4j [label="Graph Query"]
                ChromaDB -> "Phi-4 (Extract)" -> Neo4j
                Neo4j -> "Qwen/Llama (Reason)"
                "Qwen/Llama (Reason)" -> Prescription
            }
            """)
        else:
            st.graphviz_chart("""
            digraph {
                User -> Streamlit
                Streamlit -> ChromaDB [label="Similarity"]
                ChromaDB -> "Phi-4/Llama (Generate)"
                "Phi-4/Llama (Generate)" -> Answer
            }
            """)

        # Export Option - serialize UseCaseProfile for JSON
        def serialize_prescription(p):
            out = {k: v for k, v in p.items() if k != "use_case_profile"}
            out["use_case_profile"] = asdict(p["use_case_profile"])
            return json.dumps(out, indent=2)

        st.divider()
        if st.download_button(
            label="Export Prescription (JSON)",
            data=serialize_prescription(prescription),
            file_name=f"llm_prescription_{task_type.lower().replace(' ', '_')}.json",
            mime="application/json"
        ):
            st.success("Prescription saved! Use this to configure your Tier 0 deployment.")


# Streamlit pages: run when executed; skip when imported (e.g. for tests)
if __name__ == "__main__":
    render_model_evaluation_tab()
