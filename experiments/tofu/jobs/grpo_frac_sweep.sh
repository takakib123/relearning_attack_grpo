#!/usr/bin/env bash
# grpo_frac_sweep.sh
# ==================
# GRPO data-portion ablation -- the relearning-attack counterpart of
# sft_frac_sweep.sh. For each of the three forget10 Llama-3.2-1B-Instruct unlearned
# models, sweep the relearning portion (train_frac) 20/80 -> 80/20 and measure
# Q_F (attack success) vs Q_held (held-out recall) under the honest direct-match
# metric, using the dense-reward GRPO attack. 4 fracs x 3 models = 12 runs,
# pinned 2-way (cuda:0,1). Same reward/eval config as the canonical llama3.2 run
# (reward=dense, prompt_style=llama3_chat, eval=hf).
#
#   source scripts/env.sh
#   cd experiments/tofu
#   bash jobs/grpo_frac_sweep.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$HERE/src"
OUT="$HERE/results/grpo_frac_sweep"
mkdir -p "$OUT"

STEPS="${STEPS:-150}"
N_EVAL="${N_EVAL:-64}"          # match sft_frac_sweep
REWARD="${REWARD:-dense}"

DEFAULT_MODELS="simnpo|open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b4.5_a1_d1_g0.125_ep10
altpo|open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha2_epoch10
rmu|open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer5_scoeff10_epoch5"
MODELS="${MODELS:-$DEFAULT_MODELS}"

FRACS=(0.2 0.4 0.6 0.8)

# job list "tag|model|frac"
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
  ( cd "$SRC" && python grpo_attack.py \
      --unlearned_model "$model" \
      --prompt_style llama3_chat --eval hf --device "cuda:$gpu" \
      --reward "$REWARD" --train_frac "$frac" --num_outer_steps "$STEPS" \
      --n_eval "$N_EVAL" --out_dir "$OUT/$name" ) > "$OUT/$name.log" 2>&1
  echo "[gpu$gpu] DONE  $name" >&2
}
export -f run_one
export SRC OUT STEPS N_EVAL REWARD

# 2 GPUs, SLOTS runs packed per GPU (these 1B runs use ~4-15 GB and ~47% util
# each, so 2/GPU fills the card). WORKERS = NGPU*SLOTS concurrent; worker w runs
# on cuda:(w % NGPU) and processes jobs w, w+WORKERS, ...
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
