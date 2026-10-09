#!/usr/bin/env bash
# Harry Potter QA (paper Sec. 4.2.1, Tables 3 and 5).
# Target: microsoft/Llama2-7b-WhoIsHarryPotter. Needs 2 GPUs (HF training + vLLM eval)
# and an HF token with access to meta-llama/Llama-2-7b-chat-hf (export HF_TOKEN=...).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
cd experiments/harry_potter
OUT="${OUT:-results}"

# GRPO attack: 14 relearning questions, 41 held-out questions (seed 1), 50 outer steps.
# NOTE: eval_best_checkpoint=true (the script default) picks the checkpoint by mean
# held-out leakage; pass --eval_best_checkpoint false to evaluate the final model.
python grpo_hp_multi_v2.py --q_f_size 14 --seed 1 --num_outer_steps 50 \
  --checkpoint_every 10 --eval_every 10 --use_vllm_eval true \
  --train_gpu 0 --vllm_gpu 1 --log_dir "$OUT"

# Pre-unlearning reference (Llama-2-7b-chat) on the same split.
python grpo_hp_multi_v2.py --q_f_size 14 --seed 1 --base_model llama2_7b_chat \
  --eval_no_lora true --log_dir "$OUT"

# Embedding-space attack baseline (universal soft prompt, same 14/41 split).
python universal_token_sweep.py
