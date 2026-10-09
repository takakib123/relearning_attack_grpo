"""
config.py
=========
All settings for the typed-forget10 evaluation. Nothing run-affecting is hardcoded
elsewhere.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
# HERE = <repo>/experiments/tofu/src  -> REPO_ROOT is three levels up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

# --- data / outputs ---
# TOFU datasets live under data/tofu/ ; core code is on PYTHONPATH
# (see scripts/env.sh).
TYPED_JSONL = os.path.join(REPO_ROOT, "data", "tofu", "forget10_typed.jsonl")
RESULTS_DIR = os.path.join(os.path.dirname(HERE), "results")

# --- sampling / bound ---
SEED = 0
N_EVAL = 128                 # completions sampled per question (spec: 128)
ALPHA = 0.01                 # one-sided Clopper-Pearson level (matches tofu_eval)
TAUS = (0.9, 1.0)            # A-ESR thresholds for open-ended (0.9 minor var, 1.0 exact)

# --- generation ---
MAX_NEW_TOKENS = 128
TEMPERATURE = 1.0
TOP_P = 0.9
EVAL_BATCH = 64

# --- model / prompt ---
DEFAULT_MODEL = "locuslab/tofu_ft_llama2-7b"
DEFAULT_TOKENIZER = "meta-llama/Llama-2-7b-chat-hf"
PROMPT_START = "[INST] "
PROMPT_END = " [/INST]"

# --- GRPO reward redesign (grpo_attack.py; see reward-spec) ---
# Reward A (self-contrast) is a fixed function of the frozen unlearned policy's
# own conditional structure; Reward B (graded_key) is token-overlap on the manual
# deterministic keys. See validate_self_contrast.py for the pre-training gate.
REWARD_MODE = "combined"          # "graded_key" | "self_contrast" | "combined"
SC_WEIGHT = 1.0                   # weight of the standardized self-contrast term
KEY_WEIGHT = 1.0                  # weight of the standardized graded-key term
SC_LENGTH_NORMALIZE = True        # per-token r_sc (removes completion-length confound)
CONTRAST_MODE = "unconditional"   # "unconditional" (PMI, param-free) | "masked_question"
SPLIT_LEVEL = "question"          # "question" (all authors in Q_F) | "entity" (hold out authors)
N_TRAIN_AUTHORS = 10              # entity split: # authors contributing to Q_F (rest held out)

# --- vLLM backend (optional fast path; --backend vllm) ---
VLLM_GPU = 1                 # physical GPU to pin the vLLM engine to
VLLM_MEM_UTIL = 0.90
VLLM_MAX_MODEL_LEN = 1024
VLLM_ENFORCE_EAGER = False
