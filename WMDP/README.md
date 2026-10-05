# WMDP: Probabilistic Relearning Attacks

This folder has the WMDP-Deduped experiments from *Towards Probabilistic Unlearning Attacks on Large Language Models* (Section 4.2.3, Table 4, Appendix D.4). It contains the GRPO relearning attack, the SFT relearning baseline, and the data they use.

## Setting

The forget set is WMDP-Deduped: 785 biology and cybersecurity multiple-choice questions in five disjoint splits of 157 each. The attacker can see one part of the forget set, the relearning set $D_f$. The held-out evaluation set $D_e$ never enters the gradient. With the default `--skip-split 1`, split 1 is held out as $D_e$, and splits {0, 2, 3, 4} make up $D_f$.

To read an answer, we take the logits of the four letter tokens `A`–`D` right after `Answer:`. Every evaluation uses this readout:

* **Attack accuracy**: letter accuracy on $D_e$.
* **MMLU utility**: letter accuracy on 785 MMLU questions from five categories (health, history, law, philosophy, social sciences). These categories are disjoint from the WMDP domains.

The answer is a single letter token, so GRPO draws all $G$ rollouts for a prompt from one forward pass. It samples $G = 64$ letters from the softmax over the four letter logits. A sampled letter gets reward 1 if it is correct and 0 otherwise. Advantages are normalized within each group. Groups where every rollout gets the same reward carry no signal and are masked out.

| `--mode` | objective |
|---|---|
| `grpo` | group-relative policy gradient on the binary correctness reward (our attack) |
| `sft-full` | token NLL on question + choices + spelled-out correct answer (SFT relearning baseline, Deeb & Roger, 2025) |
| `sft-letter` | cross-entropy on the correct letter (ablation: same action space as `grpo`, supervised) |

All three modes share the data, prompt, optimizer (Lion), schedule and evaluation. The only difference is the objective.

## Layout

```
WMDP/
├── relearn.py              # entry point: attack + per-epoch evaluation
├── wmdp_relearn/
│   ├── data.py             # splits, MMLU retain set, prompt formats
│   ├── evaluate.py         # letter-logit MCQ accuracy
│   └── objectives.py       # grpo / sft-full / sft-letter losses
├── scripts/
│   ├── run_wmdp.sh         # exact configurations for Table 4
│   └── summarize.py        # builds Table 4 from result files
├── data/
│   ├── wmdp-deduped/split_{0..4}.jsonl
│   └── mmlu/mmlu_{health,history,law,philosophy,social_sciences}.jsonl
└── results/table4/         # the result files behind Table 4
```

Each data line is `{"question": str, "choices": [str x4], "answer": int}`.

## Setup

```bash
pip install -r requirements.txt
```

We ran the experiments on single 48 GB GPUs, using full-parameter bf16 training with gradient checkpointing. The 7B runs fit on one GPU. The TAR-Bio-v2 runs were sharded across two GPUs with `--device-map auto`.

## Models

| Unlearning method | Checkpoint | Base model (pre-unlearning) |
|---|---|---|
| SimNPO | `OPTML-Group/SimNPO-WMDP-zephyr-7b-beta` | `HuggingFaceH4/zephyr-7b-beta` |
| NPO-SAM | `OPTML-Group/NPO-SAM-WMDP` | `HuggingFaceH4/zephyr-7b-beta` |
| TAR-Bio-v2 | `lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2` | `NousResearch/Meta-Llama-3-8B-Instruct` |
| IDK-AP | `OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct` | `NousResearch/Meta-Llama-3-8B-Instruct` |

The Zephyr checkpoints and TAR-Bio-v2 use their base model's tokenizer (`--tokenizer`). IDK-AP ships its own tokenizer, which has a resized vocabulary of 128,257. Run IDK-AP on a **single GPU**: sharding it with `device_map` gives all-zero logits.

## Running

A single attack:

```bash
python relearn.py --model OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct \
    --label idk-ap --mode grpo --skip-split 1 --lr 8e-7 --batch-size 2 --epochs 8
```

Evaluation only (e.g. a pre-unlearning model):

```bash
python relearn.py --model HuggingFaceH4/zephyr-7b-beta --epochs 0
```

Reproducing Table 4 (all rows, or one row: `pre`, `simnpo`, `npo-sam`, `tar-bio-v2`, `idk-ap`):

```bash
bash scripts/run_wmdp.sh
bash scripts/run_wmdp.sh idk-ap
```

| Hyperparameter | Zephyr-7B (SimNPO, NPO-SAM) | Llama-3-8B (TAR-Bio-v2, IDK-AP) |
|---|---|---|
| learning rate (Lion) | 2e-7 | 8e-7 |
| batch size (prompts) | 4 | 2 |
| epochs / warmup steps | 8 / 24 | 8 / 24 |
| GRPO group size $G$ / temperature | 64 / 1.0 | 64 / 1.0 |

`relearn.py` writes one JSON per run. The file holds the arguments, the per-epoch history (held-out, relearning-set and MMLU accuracy, both raw and mean-calibrated), and a summary. Add `--wandb` to log training curves.

## Results

```bash
python scripts/summarize.py results/table4
```

This prints attack accuracy and MMLU utility for every checkpoint: unlearned (epoch 0), after SFT, and after GRPO. Attack accuracy is the best held-out epoch. Pass `--plateau` to use the mean of the last four epochs instead. MMLU utility is the final epoch. The **Pre** column is filled in when evaluation-only runs of the base models are in the same directory, as `run_wmdp.sh` produces.

## Acknowledgements

This code builds on the codebase of Deeb & Roger (2025), *Do Unlearning Methods Remove Information from Language Model Weights?*, at <https://github.com/aghyad-deeb/unlearning_evaluation>. The WMDP-Deduped splits, the MMLU retain subsets, the MCQ prompt format and the full-string SFT relearning objective all come from that repository. We thank the authors for releasing it. We also thank the authors of the evaluated unlearned checkpoints (SimNPO, NPO-SAM, TAR, IDK-AP) for making them public.
