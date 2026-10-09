#!/usr/bin/env bash
# Proximal GRPO relearning attack on the 80-question TOFU training set
# (data/tofu/train80.jsonl) against the forget10 SimNPO Llama-3.2-1B-Instruct
# checkpoint. Single GPU; fits in 8 GB.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../experiments/tofu_proximal"
python train.py --tag proximal_b0.05_e0.1 --beta 0.05 --eta_inv 0.1 "$@"
