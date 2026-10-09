"""
sft_attack.py
=============
Supervised-finetuning relearning BASELINE for grpo_attack.py. Directly finetunes
the unlearned model on the Q_F question->answer pairs (LoRA, capacity-matched to
the GRPO attack: r8 q_proj/v_proj), then evaluates pre->post on Q_F (attack
success) and Q_held (generalization) with the same typed direct-match metrics and
the same entity_in_qf slicing. Answers the question: how much does plain SFT on
the exact gold answers recover, versus the GRPO relearning attack?

Uses the same Q_F/Q_held split, prompt tags, and HF eval path as grpo_attack.

# Run:
#   python sft_attack.py \
#     --unlearned_model open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha1_epoch10 \
#     --prompt_style llama3_chat --device cuda:0
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import random
from datetime import date

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

import config
import helpers
from grpo_attack import UNLEARNED, build_qf_qheld, report, report_qheld
from attack.grpo_core import (attach_new_lora, build_prompt_encodings,
                              chat_template_tags)


def build_qf_qheld_frac(train_frac: float):
    """Fraction-based Q_F/Q_held split for the SFT data-portion ablation. Same
    deterministic non-yes/no pool as build_qf_qheld, but per author the first
    round(train_frac * n_author) questions (>=1) go to Q_F (the SFT training
    portion) and the rest to Q_held. All authors stay in Q_F (question-level
    split), so every Q_held row is entity_in_qf=True: this measures whether
    memorizing train_frac of an author's phrasings recovers the *held-out*
    phrasings of the same authors, as the training portion grows 20%%->80%%."""
    assert 0.0 < train_frac < 1.0, f"train_frac must be in (0,1), got {train_frac}"
    items = helpers.load_typed(config.TYPED_JSONL)
    det = [it for it in items
           if it.question_type == "deterministic" and not helpers.is_yes_no(it)]
    by_author = collections.defaultdict(list)
    for it in det:
        by_author[it.author_id].append(it)
    qf, qheld = [], []
    for aid in sorted(by_author):
        qs = sorted(by_author[aid], key=lambda it: it.idx)
        k = max(1, min(len(qs) - 1, round(train_frac * len(qs))))
        qf += qs[:k]
        qheld += qs[k:]
    return qf, qheld


def build_sft_examples(tok, items, start_tag, end_tag, add_special):
    """(input_ids, labels) per item: prompt tokens masked (-100), answer tokens
    (+ eos) supervised. Mirrors how eval prompts the model, so SFT targets the
    exact continuation the eval scores."""
    examples = []
    for it in items:
        p_ids = tok(start_tag + it.question + end_tag,
                    add_special_tokens=add_special)["input_ids"]
        a_ids = tok(" " + it.answer, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        examples.append((p_ids + a_ids, [-100] * len(p_ids) + a_ids))
    return examples


def collate(batch, pad_id, device):
    """Right-pad a batch of (input_ids, labels) for causal-LM training."""
    maxlen = max(len(ids) for ids, _ in batch)
    ii, ll, aa = [], [], []
    for ids, lab in batch:
        pad = maxlen - len(ids)
        ii.append(ids + [pad_id] * pad)
        ll.append(lab + [-100] * pad)
        aa.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(ii, device=device), torch.tensor(ll, device=device),
            torch.tensor(aa, device=device))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--unlearned_model", default=UNLEARNED)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompt_style", choices=["llama2_inst", "llama3_chat"],
                    default="llama2_inst")
    ap.add_argument("--n_per_author", type=int, default=2)
    ap.add_argument("--train_frac", type=float, default=None,
                    help="if set, split Q_F/Q_held by this per-author fraction "
                         "(SFT data-portion ablation) instead of --n_per_author")
    ap.add_argument("--split_level", choices=["question", "entity"],
                    default=config.SPLIT_LEVEL)
    ap.add_argument("--n_train_authors", type=int, default=config.N_TRAIN_AUTHORS)
    ap.add_argument("--seed", type=int, default=0)
    # SFT
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--n_eval", type=int, default=config.N_EVAL)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    if args.out_dir is None:
        args.out_dir = os.path.join(config.RESULTS_DIR, "sft_attack")
    os.makedirs(args.out_dir, exist_ok=True)
    config.N_EVAL = args.n_eval

    if args.train_frac is not None:
        qf, qh = build_qf_qheld_frac(args.train_frac)
        split_desc = f"train_frac={args.train_frac:.2f}"
    else:
        qf, qh = build_qf_qheld(args.n_per_author, args.split_level, args.n_train_authors)
        split_desc = f"split={args.split_level} n_per_author={args.n_per_author}"
    qf_author_ids = {it.author_id for it in qf}
    print(f"SFT baseline  {split_desc}: Q_F = {len(qf)}, Q_held = "
          f"{len(qh)} ({sum(it.author_id in qf_author_ids for it in qh)} in-qf-entity)",
          flush=True)

    tokenizer_id = args.tokenizer or (
        config.DEFAULT_TOKENIZER if args.prompt_style == "llama2_inst"
        else args.unlearned_model)
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    print(f"Loading {args.unlearned_model} on {args.device} + fresh LoRA "
          f"r{args.lora_rank}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.unlearned_model, dtype=torch.bfloat16, use_safetensors=True)
    base.to(args.device).eval()
    model = attach_new_lora(base, args.lora_rank, args.lora_alpha, 0.0,
                            ("q_proj", "v_proj"))
    model.print_trainable_parameters()

    if args.prompt_style == "llama3_chat":
        start_tag, end_tag = chat_template_tags(tok)
        add_special = False
    else:
        start_tag, end_tag = config.PROMPT_START, config.PROMPT_END
        add_special = True

    enc_qf = build_prompt_encodings(tok, qf, start_tag, end_tag, args.device,
                                    add_special_tokens=add_special)
    enc_qh = build_prompt_encodings(tok, qh, start_tag, end_tag, args.device,
                                    add_special_tokens=add_special)

    print("[pre] HF eval (B=0 identity) Q_F+Q_held ...", flush=True)
    pre_qf = helpers.evaluate_set(model, tok, enc_qf, config, label="pre Q_F")
    pre_qh = helpers.evaluate_set(model, tok, enc_qh, config, label="pre Q_held")

    # -------- supervised finetuning on Q_F (question -> gold answer) --------
    examples = build_sft_examples(tok, qf, start_tag, end_tag, add_special)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    train_log = open(os.path.join(args.out_dir, "train_log.csv"), "w", newline="")
    tw = csv.writer(train_log)
    tw.writerow(["epoch", "mean_loss"])
    rng = random.Random(args.seed)
    model.train()
    for epoch in range(args.epochs):
        rng.shuffle(examples)
        losses = []
        for i in range(0, len(examples), args.batch_size):
            ii, ll, aa = collate(examples[i:i + args.batch_size], tok.pad_token_id,
                                 args.device)
            loss = model(input_ids=ii, attention_mask=aa, labels=ll).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
        mean_loss = sum(losses) / len(losses)
        tw.writerow([epoch, f"{mean_loss:.5f}"])
        train_log.flush()
        print(f"[epoch {epoch:2d}] mean_loss={mean_loss:.4f}", flush=True)
    train_log.close()

    adapter_dir = os.path.join(args.out_dir, "adapter")
    model.save_pretrained(adapter_dir)
    model.eval()

    print("[post] HF eval Q_F+Q_held ...", flush=True)
    post_qf = helpers.evaluate_set(model, tok, enc_qf, config, label="post Q_F")
    post_qh = helpers.evaluate_set(model, tok, enc_qh, config, label="post Q_held")

    helpers.write_deterministic_csv(os.path.join(args.out_dir, "post_qf.csv"), post_qf)
    helpers.write_deterministic_csv(os.path.join(args.out_dir, "post_qheld.csv"), post_qh)

    print("\n" + "=" * 66 + f"\nSFT ATTACK PRE->POST ({date.today().isoformat()})  "
          f"epochs={args.epochs} {split_desc}\n" + "=" * 66)
    report("Q_F (trained, attack success)", pre_qf, post_qf)
    report_qheld(pre_qh, post_qh, qf_author_ids)
    print(f"\nCSVs + train log + adapter in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
