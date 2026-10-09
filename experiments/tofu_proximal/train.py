"""
Proximal GRPO relearning attack with a binary disclosure reward.

Outer iteration t freezes a copy pi_t of the policy and approximately solves

    pi_{t+1} = argmax_pi  p_pi - beta KL(pi || pi_0) - eta^{-1} KL(pi || pi_t)

with `--inner` gradient steps on fresh on-policy samples. For each prompt q we draw
y_1..y_G ~ pi_theta (temperature 1, no top-p / top-k, fixed length cap, capped
responses kept) and compute

    l_i  = log pi_theta(y_i|q)    l0_i = log pi_0(y_i|q)    lt_i = log pi_t(y_i|q)
    c_i  = r_i - beta (l_i - l0_i) - eta_inv (l_i - lt_i)
    g    = (1/G) sum_i stopgrad(c_i - mean_{j!=i} c_j) grad l_i

  * r_i is the binary verifier 1[y_i contains an answer key] (common.lexical_hit).
  * l, l0, lt are complete-sequence log-likelihoods (stop token included, no 1/|y|).
  * pi_0 is the released unlearned checkpoint (LoRA disabled) and never changes.
  * pi_t is a frozen copy of the LoRA weights from the start of the outer iteration.
  * The leave-one-out baseline mean_{j!=i} c_j is independent of y_i, so with
    independent on-policy samples g is an unbiased estimate of the gradient of the
    proximal objective. There is no std normalisation, no clipping and no
    importance ratio.
  * --eta_inv 0 removes the proximal term and optimises J_beta = p - beta KL(.||pi_0).
  * Groups in which every reward is 0 are kept: they still carry the KL gradient.

Optional: --retain_beta > 0 adds a KL(.||pi_0) penalty on TOFU retain prompts using
the same estimator. This extends the analysed objective, so it is off by default.
Sequence KL on retain prompts is logged after every outer iteration either way,
together with the utility bound sqrt(K_R / 2).

Usage:
  python train.py --tag proximal                  # beta = 0.05, eta_inv = 0.1
  python train.py --tag jbeta --eta_inv 0         # J_beta only
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch

from common import (DATA_DIR, RESULTS_DIR, UNLEARNED_ID, UNLEARNED_REV, dump_json, lexical_hit, load_policy,
                    load_tokenizer, read_jsonl, render_prompt, sample, seq_logprob, sha256_file,
                    tofu_retain_questions, verifier_sha256)


class LoraSnapshot:
    """Evaluates the model at frozen LoRA weights (pi_t) without keeping a second model."""

    def __init__(self, model):
        self.params = [p for n, p in model.named_parameters() if "lora_" in n]
        self.frozen = [p.detach().clone() for p in self.params]

    def refresh(self):
        for f, p in zip(self.frozen, self.params):
            f.copy_(p.detach())

    @torch.no_grad()
    def logprob(self, model, prompt_ids, comps, pad_id):
        live = [p.detach().clone() for p in self.params]
        for f, p in zip(self.frozen, self.params):
            p.data.copy_(f)
        try:
            return seq_logprob(model, prompt_ids, comps, pad_id)
        finally:
            for l, p in zip(live, self.params):
                p.data.copy_(l)

    @torch.no_grad()
    def sq_distance(self):
        return sum(((f - p.detach()) ** 2).sum().item() for f, p in zip(self.frozen, self.params))


def loo(x: torch.Tensor) -> torch.Tensor:
    """x_i - mean_{j != i} x_j."""
    return x - (x.sum() - x) / (x.numel() - 1)


def group_step(model, tok, snap, prompt_ids, keys, G, max_new, seed, beta, eta_inv, scale, grad_chunk):
    """Sample one group, backpropagate its (scaled) surrogate, return diagnostics.
    keys=None scores every response 0 (used for the optional retain penalty)."""
    pad = tok.pad_token_id
    model.eval()
    resp = sample(model, tok, prompt_ids, G, max_new, seed, batch_size=G)
    comps = [x["token_ids"] for x in resp]
    r = torch.tensor([float(lexical_hit(x["text"], keys)) if keys else 0.0 for x in resp],
                     device=next(model.parameters()).device)
    with torch.no_grad(), model.disable_adapter():
        l0 = seq_logprob(model, prompt_ids, comps, pad)
    lt = snap.logprob(model, prompt_ids, comps, pad) if eta_inv > 0 else None
    l = seq_logprob(model, prompt_ids, comps, pad)
    k0 = l - l0
    kt = (l - lt) if lt is not None else torch.zeros_like(l)
    c = r - beta * k0 - eta_inv * kt
    adv = loo(c)
    # The surrogate is linear in l_i, so it is backpropagated in chunks to bound memory.
    # train() only enables gradient checkpointing: LoRA dropout is 0, so the network
    # computes the same function as the eval-mode sampling policy.
    model.train()
    for s in range(0, G, grad_chunk):
        lg = seq_logprob(model, prompt_ids, comps[s:s + grad_chunk], pad, grad=True, chunk=grad_chunk)
        (-(adv[s:s + grad_chunk] * lg).sum() / G * scale).backward()
    model.eval()
    return {"reward": r.mean().item(), "k0": k0.mean().item(), "kt": kt.mean().item(),
            "obj": c.mean().item(), "all_zero": float(r.max().item() == 0),
            "all_one": float(r.min().item() == 1), "capped": sum(x["capped"] for x in resp) / G,
            "len": sum(len(x) for x in comps) / G, "texts": [x["text"] for x in resp],
            "rewards": r.tolist()}


@torch.no_grad()
def retain_kl(model, tok, prompts, n, max_new, seed):
    """Per-prompt Monte-Carlo estimate of K_q = E_{y~pi}[log pi(y|q) - log pi_0(y|q)]."""
    model.eval()
    ks = []
    for j, p in enumerate(prompts):
        comps = [x["token_ids"] for x in sample(model, tok, p, n, max_new, seed + j * 1000, batch_size=n)]
        l = seq_logprob(model, p, comps, tok.pad_token_id)
        with model.disable_adapter():
            l0 = seq_logprob(model, p, comps, tok.pad_token_id)
        ks.append((l - l0).mean().item())
    return ks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="run name; outputs go to results/<tag>/")
    ap.add_argument("--data", default=str(DATA_DIR / "train80.jsonl"))
    ap.add_argument("--G", type=int, default=16, help="responses per prompt")
    ap.add_argument("--prompts_per_step", type=int, default=4)
    ap.add_argument("--outer", type=int, default=20, help="outer proximal iterations")
    ap.add_argument("--inner", type=int, default=10, help="gradient steps per outer iteration")
    ap.add_argument("--beta", type=float, default=0.05, help="weight on KL(pi || pi_0)")
    ap.add_argument("--eta_inv", type=float, default=0.1, help="weight on KL(pi || pi_t); 0 = J_beta only")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--grad_clip", type=float, default=0.0, help="0 = off; clipping changes the update")
    ap.add_argument("--grad_chunk", type=int, default=4, help="responses per backward pass (memory only)")
    ap.add_argument("--retain_beta", type=float, default=0.0)
    ap.add_argument("--retain_prompts", type=int, default=32)
    ap.add_argument("--retain_eval_n", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=5, help="save the adapter every k outer iterations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert args.G >= 2 and args.beta >= 0 and args.eta_inv >= 0 and args.retain_beta >= 0

    out = RESULTS_DIR / args.tag
    out.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.data)
    assert len({r["idx"] for r in rows}) == len(rows)
    by_idx = {r["idx"]: r for r in rows}
    tok = load_tokenizer()
    # LoRA dropout must be 0, otherwise the trained forward is not the sampling policy.
    model = load_policy(dict(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    prompts = {i: render_prompt(tok, r["question"]) for i, r in by_idx.items()}
    retain = tofu_retain_questions(args.retain_prompts, seed=args.seed)
    retain_ids = [render_prompt(tok, q["question"]) for q in retain]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    snap = LoraSnapshot(model)

    dump_json(out / "config.json", {
        **vars(args), "data": Path(args.data).name, "data_sha256": sha256_file(args.data),
        "pi_0": {"model_id": UNLEARNED_ID, "revision": UNLEARNED_REV},
        "verifier_sha256": verifier_sha256(),
        "decoding": "temperature 1, top_p 1, top_k 0, fixed cap, capped responses kept",
        "retain_idx": [q["retain_idx"] for q in retain],
        "trainable_params": sum(p.numel() for p in params)})
    log_f = (out / "steps.jsonl").open("a")
    roll_f = (out / "rollouts.jsonl").open("a")

    k_start = retain_kl(model, tok, retain_ids, args.retain_eval_n, args.max_new_tokens, 10 ** 7)
    order, step, t0 = [], 0, time.time()
    for t in range(args.outer):
        snap.refresh()                                  # pi_t := current policy
        for inner in range(args.inner):
            opt.zero_grad(set_to_none=True)
            stats = []
            for _ in range(args.prompts_per_step):
                if not order:
                    order = list(by_idx)
                    random.shuffle(order)
                qi = order.pop()
                s = group_step(model, tok, snap, prompts[qi], by_idx[qi]["answer_keys"], args.G,
                               args.max_new_tokens, args.seed + 7919 * step + qi, args.beta, args.eta_inv,
                               1.0 / args.prompts_per_step, args.grad_chunk)
                roll_f.write(json.dumps({"step": step, "outer": t, "idx": qi, "rewards": s.pop("rewards"),
                                         "texts": s.pop("texts"), "k0": s["k0"]}, ensure_ascii=False) + "\n")
                stats.append({"idx": qi, **s})
            if args.retain_beta > 0:
                for j in random.sample(range(len(retain_ids)), args.prompts_per_step):
                    group_step(model, tok, snap, retain_ids[j], None, args.G, args.max_new_tokens,
                               args.seed + 104729 * step + j, args.retain_beta, args.eta_inv,
                               1.0 / args.prompts_per_step, args.grad_chunk)
            gnorm = torch.nn.utils.clip_grad_norm_(
                params, args.grad_clip if args.grad_clip > 0 else float("inf")).item()
            opt.step()
            rec = {"step": step, "outer": t, "inner": inner, "grad_norm": gnorm,
                   "wall_s": round(time.time() - t0, 1),
                   **{k: sum(s[k] for s in stats) / len(stats)
                      for k in ["reward", "k0", "kt", "obj", "all_zero", "all_one", "capped", "len"]},
                   "per_prompt": [{"idx": s["idx"], "reward": s["reward"], "k0": s["k0"]} for s in stats]}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            roll_f.flush()
            print(f"[t={t:2d} k={inner:2d}] reward={rec['reward']:.3f} K0={rec['k0']:+.3f} "
                  f"Kt={rec['kt']:+.3f} obj={rec['obj']:+.3f} all_zero={rec['all_zero']:.2f} "
                  f"|g|={gnorm:.3f} {rec['wall_s']}s", flush=True)
            step += 1
        k_ret = retain_kl(model, tok, retain_ids, args.retain_eval_n, args.max_new_tokens, 10 ** 7 + t + 1)
        m_ret = sum(k_ret) / len(k_ret)
        log_f.write(json.dumps({"outer_end": t, "lora_sq_dist_to_pi_t": snap.sq_distance(),
                                "retain_K_mean": m_ret, "retain_K_at_start": sum(k_start) / len(k_start),
                                "utility_bound": max(m_ret, 0.0) ** 0.5 / 2 ** 0.5}) + "\n")
        log_f.flush()
        print(f"== outer {t} done: retain K = {m_ret:+.4f}, |dU| <= {max(m_ret, 0) ** .5 / 2 ** .5:.3f}",
              flush=True)
        if (t + 1) % args.save_every == 0 or t + 1 == args.outer:
            model.save_pretrained(out / f"adapter_outer{t + 1:03d}")
    log_f.close()
    roll_f.close()


if __name__ == "__main__":
    main()
