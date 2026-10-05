"""Relearning attacks on WMDP-unlearned models (GRPO and SFT baselines).

The forget set (WMDP-Deduped, 5 x 157 MCQs) is split into a relearning set D_f
available to the attacker and a disjoint held-out evaluation set D_e that never
enters the gradient. Each epoch we report letter-logit accuracy on D_e (attack
accuracy), on D_f, and on 785 MMLU questions (retained utility).

Usage:
    python relearn.py --model OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct \
        --mode grpo --skip-split 1 --lr 8e-7

    # evaluation only (e.g. the pre-unlearning model)
    python relearn.py --model NousResearch/Meta-Llama-3-8B-Instruct --epochs 0

See scripts/run_wmdp.sh for the exact configurations behind the paper's table.
"""
import argparse
import json
import os
import random

# Reduces allocator fragmentation; needed to full-finetune an 8B model with Lion
# on a single ~48GB GPU. Must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from lion_pytorch import Lion
from transformers import AutoModelForCausalLM, AutoTokenizer

from wmdp_relearn.data import (
    DATA_DIR, NUM_SPLITS, load_mmlu_retain, load_wmdp_splits,
)
from wmdp_relearn.evaluate import evaluate, letter_logits, letter_token_ids
from wmdp_relearn.objectives import grpo_loss, sft_full_loss, sft_letter_loss


class Logger:
    """Thin wrapper so Weights & Biases is optional."""

    def __init__(self, enabled, **init_kwargs):
        self.run = None
        if enabled:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(**init_kwargs)
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch")

    def log(self, d):
        if self.run is not None:
            self.wandb.log(d)

    def finish(self, summary):
        if self.run is not None:
            self.wandb.summary.update(summary)
            self.wandb.finish()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="HF id or path of the attacked model")
    ap.add_argument("--tokenizer", default=None, help="defaults to --model")
    ap.add_argument("--label", default=None, help="name used for the output file")
    ap.add_argument("--mode", choices=["grpo", "sft-full", "sft-letter"], default="grpo")
    ap.add_argument("--skip-split", type=int, default=1,
                    help="WMDP split held out as D_e; the other four form D_f")
    ap.add_argument("--skip-splits", type=int, nargs="+", default=None,
                    help="hold out several splits (overrides --skip-split), "
                         "e.g. '--skip-splits 1 3' trains on {0,2,4}")
    ap.add_argument("--group-size", type=int, default=64, help="GRPO rollouts per prompt (G)")
    ap.add_argument("--temperature", type=float, default=1.0, help="GRPO sampling temperature")
    ap.add_argument("--entropy-coeff", type=float, default=0.0, help="GRPO entropy bonus")
    ap.add_argument("--lr", type=float, default=8e-7)
    ap.add_argument("--epochs", type=int, default=8, help="0 = evaluate only")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--val-batch-size", type=int, default=4)
    ap.add_argument("--warmup-steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device-map", default=None,
                    help='e.g. "auto" to shard the model across visible GPUs. Do not '
                         'use for checkpoints with a resized vocabulary (IDK-AP).')
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default="results")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    ap.add_argument("--wandb-project", default="wmdp-relearn")
    return ap.parse_args()


def main():
    args = parse_args()
    label = args.label or os.path.basename(args.model.rstrip("/"))
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    held_splits = sorted(set(
        args.skip_splits if args.skip_splits is not None else [args.skip_split]
    ))
    train_splits = [i for i in range(NUM_SPLITS) if i not in held_splits]
    skip_tag = "skip" + "_".join(str(s) for s in held_splits)
    run_name = f"{label}-{args.mode}-{skip_tag}-lr{args.lr:g}"
    logger = Logger(args.wandb, project=args.wandb_project, name=run_name,
                    config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    label_ids = letter_token_ids(tokenizer)

    if args.device_map:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map=args.device_map,
        )
        # inputs go to the device holding the embedding layer
        device = next(model.parameters()).device
    else:
        device = torch.device("cuda")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    optimizer = Lion(model.parameters(), lr=args.lr, use_triton=True)

    train_set = load_wmdp_splits(train_splits, args.data_dir)   # D_f
    heldout = load_wmdp_splits(held_splits, args.data_dir)      # D_e
    retain = load_mmlu_retain(args.data_dir)

    print(f"===== {args.mode} attack on {label} =====")
    print(f"train splits {train_splits} (n={len(train_set)}), "
          f"held out {held_splits} (n={len(heldout)}), retain n={len(retain)}")

    hist = []

    def run_eval(epoch):
        # free the previous epoch's gradients before the eval forward passes
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        model.eval()
        ho_raw, ho_cal = evaluate(model, tokenizer, device, heldout, label_ids,
                                  args.val_batch_size)
        tr_raw, _ = evaluate(model, tokenizer, device, train_set[:400], label_ids,
                             args.val_batch_size)
        re_raw, re_cal = evaluate(model, tokenizer, device, retain, label_ids,
                                  args.val_batch_size)
        rec = {"epoch": epoch, "heldout_raw": ho_raw, "heldout_cal": ho_cal,
               "trained_raw": tr_raw, "retain_raw": re_raw, "retain_cal": re_cal}
        logger.log({"eval/epoch": epoch,
                    **{f"eval/{k}": v for k, v in rec.items() if k != "epoch"}})
        hist.append(rec)
        print(f"[ep{epoch}] held-out {ho_raw:.3f} (cal {ho_cal:.3f}) | "
              f"trained-splits {tr_raw:.3f} | retain {re_raw:.3f} (cal {re_cal:.3f})")
        model.train()
        return rec

    run_eval(0)

    step = 0
    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(train_set)
        ep_reward, ep_alive, ep_prompts, ep_grad = [], 0, 0, []
        for i in range(0, len(train_set), args.batch_size):
            batch = train_set[i : i + args.batch_size]
            step += 1
            lr = args.lr * max(0, min(1, step / args.warmup_steps))
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.zero_grad()
            answers = torch.tensor([p["answer"] for p in batch], device=device)

            if args.mode == "sft-full":
                loss, stats = sft_full_loss(model, tokenizer, batch, device)
            elif args.mode == "sft-letter":
                lg = letter_logits(model, tokenizer, batch, device, label_ids)
                loss, stats = sft_letter_loss(lg, answers)
            else:
                lg = letter_logits(model, tokenizer, batch, device, label_ids)
                loss, stats = grpo_loss(lg, answers, args.group_size,
                                        args.temperature, args.entropy_coeff)
                ep_reward.append(stats["mean_reward"])
                ep_alive += stats.pop("n_alive")
            ep_prompts += len(batch)

            if loss is None:  # GRPO: every group degenerate, no gradient
                logger.log({"train/step": step, "train/lr": lr, "train/grad_norm": 0.0,
                            **{f"train/{k}": v for k, v in stats.items()}})
                continue

            loss.backward()
            # global grad norm; an infinite max_norm computes it without clipping
            gnorm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf")))
            optimizer.step()
            ep_grad.append(gnorm)
            logger.log({"train/step": step, "train/lr": lr, "train/loss": loss.item(),
                        "train/grad_norm": gnorm,
                        **{f"train/{k}": v for k, v in stats.items()}})

        rec = {"mean_grad_norm": float(np.mean(ep_grad)) if ep_grad else 0.0}
        if args.mode == "grpo":
            rec["mean_reward"] = float(np.mean(ep_reward))
            rec["alive_frac"] = ep_alive / max(ep_prompts, 1)
            print(f"  epoch {epoch}: mean reward {rec['mean_reward']:.3f} | "
                  f"non-degenerate groups {ep_alive}/{ep_prompts} "
                  f"({rec['alive_frac']:.1%})")
        else:
            print(f"  epoch {epoch}: {args.mode}, {ep_prompts} prompts")
        run_eval(epoch + 1).update(rec)

    base, final = hist[0], hist[-1]
    best = max(hist, key=lambda r: r["heldout_raw"])
    summary = {
        "heldout_start": base["heldout_raw"],
        "heldout_best": best["heldout_raw"],
        "heldout_best_epoch": best["epoch"],
        "heldout_final": final["heldout_raw"],
        # mean of the last 4 evaluations; less noisy than the best epoch
        "heldout_plateau_last4": float(np.mean([r["heldout_raw"] for r in hist[-4:]])),
        "retain_start": base["retain_raw"],
        "retain_final": final["retain_raw"],
        "trained_final": final["trained_raw"],
    }
    print(f"\nheld-out: {base['heldout_raw']:.3f} -> best {best['heldout_raw']:.3f} "
          f"(ep{best['epoch']}), final {final['heldout_raw']:.3f} | "
          f"retain {base['retain_raw']:.3f} -> {final['retain_raw']:.3f}")

    os.makedirs(args.out, exist_ok=True)
    tag = f"{label}-{args.mode}-{skip_tag}" if args.epochs > 0 else f"{label}-eval-{skip_tag}"
    if args.mode == "grpo" and args.epochs > 0:
        tag += (f"-lr{args.lr:g}-T{args.temperature:g}-ent{args.entropy_coeff:g}"
                f"-G{args.group_size}-ep{args.epochs}")
    path = os.path.join(args.out, f"{tag}.json")
    with open(path, "w") as f:
        json.dump({"model": args.model, "args": vars(args), "history": hist,
                   "summary": summary}, f, indent=2)
    print(f"wrote {path}")
    logger.finish(summary)


if __name__ == "__main__":
    main()
