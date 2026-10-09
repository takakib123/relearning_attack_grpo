#!/usr/bin/env bash
# sft_frac_sweep.sh
# =================
# SFT data-portion ablation: for each of the three forget10 Llama-3.2-1B-Instruct
# unlearned models, sweep the SFT training portion (train_frac) from 20/80 up to
# 80/20 and measure Q_F (memorization / attack success) vs Q_held (held-out
# recall of the SAME authors' other phrasings) under the honest direct-match
# metric. 4 fracs {20,40,60,80} x 3 models = 12 SFT runs, pinned 2-way (cuda:0,1).
#
#   source scripts/env.sh
#   cd experiments/tofu
#   bash jobs/sft_frac_sweep.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # experiments/tofu
SRC="$HERE/src"
OUT="$HERE/results/sft_frac_sweep"
mkdir -p "$OUT"

EPOCHS="${EPOCHS:-60}"
N_EVAL="${N_EVAL:-64}"   # 64 samples/question: ~0.016 resolution, ~4h total on 3 GPUs

# model list: "tag|hf_id" per line. Override MODELS to target other checkpoints.
DEFAULT_MODELS="simnpo|open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b4.5_a1_d1_g0.125_ep10
altpo|open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha2_epoch10
rmu|open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer5_scoeff10_epoch5"
MODELS="${MODELS:-$DEFAULT_MODELS}"

FRACS=(0.2 0.4 0.6 0.8)

# Build the job list: "tag|model|frac"
JOBS=()
for f in "${FRACS[@]}"; do
  while IFS= read -r ml; do
    [ -z "$ml" ] && continue
    JOBS+=("${ml%%|*}|${ml#*|}|$f")
  done <<< "$MODELS"
done

run_one() {
  local spec="$1" gpu="$2"
  IFS='|' read -r tag model frac <<< "$spec"
  local pct; pct="$(printf '%02.0f' "$(echo "$frac * 100" | bc -l)")"
  local name="${tag}_f${pct}"
  echo "[gpu$gpu] START $name (train_frac=$frac)" >&2
  ( cd "$SRC" && python sft_attack.py \
      --unlearned_model "$model" \
      --prompt_style llama3_chat --device "cuda:$gpu" \
      --train_frac "$frac" --epochs "$EPOCHS" --n_eval "$N_EVAL" \
      --out_dir "$OUT/$name" ) > "$OUT/$name.log" 2>&1
  echo "[gpu$gpu] DONE  $name" >&2
}
export -f run_one
export SRC OUT EPOCHS N_EVAL

# 2 GPUs, SLOTS runs packed per GPU (eval-heavy 1B runs sit ~47% util each, so
# 2/GPU fills the card). WORKERS = NGPU*SLOTS; worker w -> cuda:(w % NGPU).
NGPU=2
SLOTS="${SLOTS:-2}"
WORKERS=$((NGPU * SLOTS))
for w in $(seq 0 $((WORKERS - 1))); do
  g=$((w % NGPU))
  (
    i=$w
    while [ $i -lt ${#JOBS[@]} ]; do
      run_one "${JOBS[$i]}" "$g"
      i=$((i + WORKERS))
    done
  ) &
done
wait
echo "ALL DONE -> $OUT"
