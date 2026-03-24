"""
Configuration for Mutation-Informalize Pipeline (NEW ARCHITECTURE).
Phase 1: Validate + Classify + Fix SMT
Phase 2: Mutate SMT (Pure Algorithm with Smart Retry)
Phase 3: Informalize (Agent)
Phase 4: Merge
"""
import os

# ─── NVIDIA API (for Phase 1 & 3) ───
# Set NVIDIA_API_KEYS in your environment as a comma-separated list (or via a .env file)
NVIDIA_API_KEYS = [k.strip() for k in os.environ.get("NVIDIA_API_KEYS", "").split(",") if k.strip()]
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "openai/gpt-oss-120b"

# Rate limiting (per key)
RPM_PER_KEY = 20
MAX_CONCURRENT = len(NVIDIA_API_KEYS) * RPM_PER_KEY  # 100 total
DELAY_BETWEEN = 60 / RPM_PER_KEY  # 3s per key
MAX_API_RETRIES = 5
BACKOFF_BASE = 8
MAX_TOKENS = 4096

# ─── Paths ───
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "finetune", "mine")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "finetune", "augmented")

# NEW INPUT: train_vietnamese_final.jsonl (14.87K items)
INPUT_FILE = os.path.join(DATA_DIR, "train_vietnamese_final.jsonl")

# Phase outputs (all resumeable via JSONL checkpoint)
PHASE1_VALIDATED_OUTPUT = os.path.join(OUTPUT_DIR, "phase1_validated_complete.jsonl")
PHASE2_MUTATED_OUTPUT = os.path.join(OUTPUT_DIR, "phase2_mutated_smt.jsonl")
PHASE3_INFORMALIZED_OUTPUT = os.path.join(OUTPUT_DIR, "phase3_informalized_clean.jsonl")
PHASE4_FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "train_augmented_final.jsonl")

# ─── Phase 2: Mutation Strategy Config ───
# Each strategy has different retry parameters
MUTATION_STRATEGIES = {
    "constant_variation": {
        "name": "Constant Variation",
        "initial_variations": [0.10, 0.20, -0.10, -0.20],  # ±10%, ±20%
        "retry_variations": [0.05, 0.02, -0.05, -0.02],   # ±5%, ±2% (retry)
        "level_adjust": 0,  # Keep same level
    },
    "operator_modification": {
        "name": "Operator Modification",
        "operators": {"+": "-", "-": "+", "*": "/", "/": "*"},
        "epsilon_adjustments": [0.1, 0.05, 0.01],  # Epsilon tweaks for retry
        "level_adjust": 1,  # +1 level
    },
    "variable_substitution": {
        "name": "Variable Substitution + Nesting",
        "max_nesting": 2,
        "level_adjust": 2,  # +2 levels (capped at 5)
    },
}

# Skip arithmetic subjects (already ~7.47K items)
SKIP_SUBJECTS = {"arithmetic"}

# Z3 timeout (ms)
Z3_TIMEOUT = 5000

# ─── Ensure output dir exists ───
os.makedirs(OUTPUT_DIR, exist_ok=True)
