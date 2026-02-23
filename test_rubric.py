# test_rubric.py (save in root, run once)
import sys
import importlib.util

# Load pages/2_Model_Evaluation.py (module name can't start with digit)
spec = importlib.util.spec_from_file_location(
    "model_evaluation",
    "pages/2_Model_Evaluation.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["model_evaluation"] = mod
spec.loader.exec_module(mod)

LLMPrescriptionRubric = mod.LLMPrescriptionRubric
UseCaseProfile = mod.UseCaseProfile

# Test case: High-compliance financial analysis
profile = UseCaseProfile(
    task_type="Analysis",
    data_volume="Medium (1K-10K)",
    reasoning_depth="Multi-hop",
    latency_requirement="Fast (<5s)",
    accuracy_criticality="Compliance/Legal",
    vram_available=24,
    multi_document=True,
    relationship_complexity="Complex Graph"
)

rubric = LLMPrescriptionRubric()
rx = rubric.evaluate(profile)

assert rx['primary_model'] in ['qwen14b', 'llama3.1-8b'], "Should prescribe reasoning model"
assert rx['recommended_architecture'] == "GraphRAG", "Should trigger GraphRAG for multi-doc"
assert rx['primary_details']['vram_gb'] <= 24 * 0.8, "Should respect VRAM buffer"
print("Rubric logic validated:", rx['primary_model'])
