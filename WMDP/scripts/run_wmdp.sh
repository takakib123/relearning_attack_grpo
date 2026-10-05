#!/usr/bin/env bash
# Reproduces the WMDP-Deduped table (Table 4): pre-unlearning reference, then the
# GRPO attack and the SFT baseline on each unlearned checkpoint.
#
# Split 1 is held out (D_e, 157 MCQs); splits {0,2,3,4} form D_f.
# 7B (Zephyr) runs: lr 2e-7, batch 4.  8B (Llama-3) runs: lr 8e-7, batch 2.
# All runs: 8 epochs, Lion, 24 warmup steps, G=64, temperature 1, seed 0.
#
#   bash scripts/run_wmdp.sh            # everything
#   bash scripts/run_wmdp.sh idk-ap     # one row
set -euo pipefail
cd "$(dirname "$0")/.."

ONLY=${1:-all}
OUT=${OUT:-results/runs}
PY=${PY:-python}
COMMON="--skip-split 1 --epochs 8 --group-size 64 --temperature 1.0 --seed 0 --out $OUT"

ZEPHYR=HuggingFaceH4/zephyr-7b-beta
LLAMA3=NousResearch/Meta-Llama-3-8B-Instruct   # mirror of meta-llama/Meta-Llama-3-8B-Instruct

run() {
  local tag=$1; shift
  if [[ "$ONLY" == all || "$ONLY" == "$tag" ]]; then "$PY" -u relearn.py "$@"; fi
}

# Pre-unlearning reference (evaluation only)
run pre --model $ZEPHYR --label zephyr-7b-beta --epochs 0 --val-batch-size 8 --out $OUT
run pre --model $LLAMA3 --label llama3-8b-instruct --epochs 0 --val-batch-size 4 --out $OUT

for MODE in grpo sft-full; do
  # SimNPO and NPO-SAM on Zephyr-7B-beta
  run simnpo  --model OPTML-Group/SimNPO-WMDP-zephyr-7b-beta --tokenizer $ZEPHYR \
      --label simnpo --mode $MODE --lr 2e-7 --batch-size 4 --val-batch-size 8 $COMMON
  run npo-sam --model OPTML-Group/NPO-SAM-WMDP --tokenizer $ZEPHYR \
      --label npo-sam --mode $MODE --lr 2e-7 --batch-size 4 --val-batch-size 8 $COMMON

  # TAR-Bio-v2 on Llama-3-8B-Instruct (sharded over the visible GPUs)
  run tar-bio-v2 --model lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2 --tokenizer $LLAMA3 \
      --label tar-bio-v2 --mode $MODE --lr 8e-7 --batch-size 2 --val-batch-size 8 \
      --device-map auto $COMMON

  # IDK-AP on Llama-3-8B-Instruct. Single GPU only: the checkpoint has a resized
  # vocabulary (128257) and device_map sharding yields all-zero logits.
  run idk-ap --model OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct \
      --label idk-ap --mode $MODE --lr 8e-7 --batch-size 2 --val-batch-size 4 $COMMON
done

"$PY" scripts/summarize.py "$OUT"
