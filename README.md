# Towards Probabilistic Unlearning Attacks on Large Language Models

Code for the GRPO-based probabilistic relearning attack. The attack fine-tunes a released
unlearned model with LoRA so that sampled responses disclose a target fact more often. It
rewards any sampled response that contains the target answer and penalizes KL divergence
from the released model.

This repository covers the Harry Potter QA and TOFU experiments. The WMDP experiments are
released separately.

## Layout

```
src/
  attack/grpo_core.py        GRPO rollouts, clipped surrogate + per-token KL to the released model,
                             keyword reward, sampled leakage evaluation
  attack/grpo_vllm_eval.py   batched vLLM evaluation (LoRA hot-swap)
  attack/extraction.py       extraction-strength reward term used by the TOFU attack
  attack/unlearning_utils.py helpers for the embedding-space baseline
  oracle/tofu_oracle.py      stem-tolerant keyword matching, ROUGE-L recall
  leakage_bounds.py          two-sided Clopper-Pearson leakage bounds (eq. 5) and Leak@k
experiments/
  harry_potter/              GRPO attack + embedding-space baseline on Llama2-7b-WhoIsHarryPotter
  tofu/                      SFT and GRPO relearning sweeps on TOFU forget10 (Llama-3.2-1B-Instruct)
  tofu_proximal/             proximal GRPO with a binary reward, leave-one-out baseline and
                             complete-sequence KL to the released model (single GPU)
data/tofu/                   forget10 questions typed as deterministic/open-ended with answer keys,
                             the 20 birth-city probes, and the 80-question proximal-GRPO training set
scripts/                     env.sh and end-to-end run scripts
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
source scripts/env.sh                 # adds src/ to PYTHONPATH
export HF_TOKEN=...                   # needed for the gated Llama-2 tokenizer
```

The experiments use two GPUs, one for training and one for the vLLM evaluation engine. The
7B Harry Potter runs need 80 GB cards.

## Harry Potter QA (Section 4.2.1)

```bash
bash scripts/run_hp.sh
```

This script runs the following:

- **GRPO attack**: [experiments/harry_potter/grpo_hp_multi_v2.py](experiments/harry_potter/grpo_hp_multi_v2.py).
  It splits the 55 questions into a relearning set and a held-out set using a fixed seed,
  trains a rank-8 LoRA on `q_proj`/`v_proj`, and evaluates both sets before and after the
  attack. Each question gets 128 samples at T=1.0 and top-p=0.9. Every config field is
  also a CLI flag (`--help`).
- **Pre-unlearning reference**: the same script run with
  `--base_model llama2_7b_chat --eval_no_lora true`.
- **Embedding-space baseline**: [experiments/harry_potter/universal_token_sweep.py](experiments/harry_potter/universal_token_sweep.py).
  It optimizes a universal soft prompt on the relearning questions and evaluates it on the
  held-out ones.

Each evaluation writes one row per question with `s_n` (matches out of `n_samples`),
`p_hat`, and a greedy-leak flag. To turn match counts into the per-question 99% intervals
reported in the paper, run:

```bash
python src/leakage_bounds.py --csv <eval_csv>
```

The per-question outputs behind the paper's HP tables are in
[experiments/harry_potter/paper_results/](experiments/harry_potter/paper_results/).

## TOFU (Section 4.2.2, Appendix D.3.1)

```bash
bash scripts/run_tofu.sh
```

This runs the original-question protocol. For each unlearned checkpoint (SimNPO, AltPO,
RMU) and each relearning portion in {20, 40, 60, 80}%, it trains on that share of every
author's deterministic questions and evaluates on the rest:

- **SFT**: [experiments/tofu/src/sft_attack.py](experiments/tofu/src/sft_attack.py)
- **GRPO**: [experiments/tofu/src/grpo_attack.py](experiments/tofu/src/grpo_attack.py) with `--reward dense`

`jobs/collect_sweep.py` collects the results, and `jobs/plot_figure4.py` draws Figure 4.
Called with no arguments, `plot_figure4.py` redraws the figure from the bundled numbers in
[experiments/tofu/paper_results/figure4/](experiments/tofu/paper_results/figure4/).

To evaluate any checkpoint on the typed forget10 set without an attack, run:

```bash
cd experiments/tofu/src
python main.py --model <hf_model_id> --tag <name>
```

## TOFU: proximal GRPO

```bash
bash scripts/run_tofu_proximal.sh          # or: cd experiments/tofu_proximal && python train.py --help
```

[experiments/tofu_proximal/train.py](experiments/tofu_proximal/train.py) trains a rank-8 LoRA on the
forget10 SimNPO Llama-3.2-1B-Instruct checkpoint (π₀) to optimise the objective analysed in the
paper, rather than the clipped GRPO surrogate. Outer iteration t freezes a copy π_t of the policy
and takes `--inner` steps on fresh on-policy samples toward

    argmax_π  p_π − β KL(π ‖ π₀) − η⁻¹ KL(π ‖ π_t).

For each question it samples G responses. It sets c_i = r_i − β(ℓ_i − ℓ₀ᵢ) − η⁻¹(ℓ_i − ℓ_tᵢ) and
follows the gradient (1/G) Σ_i stopgrad(c_i − mean_{j≠i} c_j) ∇ℓ_i, where the ℓ are
complete-sequence log-likelihoods under π_θ, π₀ and π_t. Specifically:

- **Reward.** Binary: 1 if a response contains an answer key as a whole-word phrase after NFKC
  normalisation and case folding, else 0. The verifier is fixed before training, and its source
  hash is stored in each run's `config.json`.
- **Advantages.** Leave-one-out baseline, with no standard-deviation scaling, no ratio clipping
  and no importance weights. Groups with all-zero reward are kept.
- **KL.** Summed over every completion token, including the stop token. It is never divided by
  response length, and π₀ is never updated.
- **Decoding.** Temperature 1 with no top-p or top-k truncation and a 64-token cap. Responses
  that hit the cap are kept.
- **Variants.** `--eta_inv 0` optimises J_β = p − β KL(·‖π₀) without the proximal term.
  `--retain_beta` adds an optional KL penalty on TOFU retain prompts, which extends the objective.

**Data.** [data/tofu/train80.jsonl](data/tofu/train80.jsonl) holds 80 deterministic forget10
questions that never mention an author's birth city. They are the questions on which a
GRPO-attacked model disclosed the answer most often over 128 samples each.

**Outputs.** Each run writes to `experiments/tofu_proximal/results/<tag>/`:

- `steps.jsonl`: per step, the mean reward, the KL estimates to π₀ and π_t, the objective, the
  share of all-zero groups, the share of capped responses and the gradient norm. After each outer
  iteration it also logs the retain-prompt KL and the utility bound √(K_R/2).
- `rollouts.jsonl`: every sampled response and its reward.
- `adapter_outer*/`: the LoRA adapters.

With the defaults (G = 16, 4 questions per step, 20 × 10 steps), a step takes about 7 s on one
8 GB GPU.

## Evaluation protocol

- **Leakage**: a sampled response leaks if it contains a target keyword. The Harry Potter
  experiments use a case-insensitive substring match. TOFU uses a stem-tolerant contiguous
  token match and discards degenerate (repetitive) completions.
- **Bounds**: per-question Clopper-Pearson intervals at α = 0.01. These hold for each
  question separately, not simultaneously across questions.

## Citation

```bibtex
@inproceedings{anonymous2027probabilistic,
  title     = {Towards Probabilistic Unlearning Attacks on Large Language Models},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2027}
}
```
