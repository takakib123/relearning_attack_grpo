#!/usr/bin/env bash
# TOFU original-question protocol (paper Appendix D.3.1, Figure 4 and Table 7).
# SFT and GRPO relearning sweeps over the relearning portion {20,40,60,80}% for the
# SimNPO, AltPO and RMU forget10 Llama-3.2-1B-Instruct checkpoints. Uses 2 GPUs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
cd experiments/tofu
bash jobs/sft_frac_sweep.sh
bash jobs/grpo_frac_sweep.sh
python jobs/collect_sweep.py sft_frac_sweep
python jobs/collect_sweep.py grpo_frac_sweep
python jobs/plot_figure4.py --sft results/sft_frac_sweep/summary.csv \
                            --grpo results/grpo_frac_sweep/summary.csv
