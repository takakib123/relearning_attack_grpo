"""
grpo_attack.py
==============
GRPO relearning attack on the SimNPO forget10 unlearned model, using a Q_F of
TWO deterministic (fact-recall) questions per author (2 x 20 = 40). Trains a
fresh LoRA on Q_F only with a direct-match reward (reward = 1 iff the completion
contains the verified answer_key), then evaluates pre->post on Q_F (attack
success) and Q_held (generalization to the author's other deterministic facts).

Reuses:
  - training core: shared.grpo_core (attach_new_lora, sample_rollouts,
    policy_forward_with_kl, grpo_loss, build_prompt_encodings)
  - reward primitives: tofu_oracle (keyword_hit, is_degenerate)
  - eval + scoring: evaluation.helpers (evaluate_set_vllm, score_item, ...)

Pre-eval reuse: a freshly-attached LoRA with B=0 is identity, so the pre-attack
reference IS the unlearned model already evaluated in
results/eval_unlearned_vllm_deterministic.csv -- we slice it, no re-run.

Training runs on --device (HF); post-eval runs on --vllm_gpu (vLLM + LoRA adapter).

# Run: python grpo_attack.py --device cuda:0 --vllm_gpu 1
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

import config
import helpers
from attack.grpo_core import (
    attach_new_lora, build_prompt_encodings, chat_template_tags, grpo_loss,
    policy_forward_with_kl, sample_rollouts, self_contrast_reward,
)
from oracle.tofu_oracle import keyword_hit, is_degenerate, _stem_tokens
from attack.extraction import es_rollout

UNLEARNED = "OPTML-Group/SimNPO-TOFU-forget10-Llama-2-7b-chat"
PRE_CSV = os.path.join(config.RESULTS_DIR, "eval_unlearned_vllm_deterministic.csv")


@dataclass
class Cfg:
    group_size: int = 8
    max_new_tokens: int = 128
    sampling_temperature: float = 1.0
    sampling_top_p: float = 0.9


def direct_match(text: str, item) -> float:
    """1.0 iff the completion contains the answer_key (degenerate text -> 0)."""
    if is_degenerate(text):
        return 0.0
    return 1.0 if keyword_hit(text, item.answer_keys) else 0.0


def make_reward(mode: str, es_weight: float, tok):
    """Return reward(text, item). 'direct' = binary answer_key match. 'dense' =
    (1-w)*direct_match + w*extraction_strength; the continuous ES term gives
    partial credit to near-miss rollouts so within-group reward variance survives
    (binary+G=8 otherwise saturates, killing the GRPO advantage signal)."""
    if mode == "direct":
        return lambda text, item: direct_match(text, item)

    def dense(text, item):
        dm = direct_match(text, item)
        if is_degenerate(text):
            return 0.0
        esr = es_rollout(text, item.answer, tok)
        return (1.0 - es_weight) * dm + es_weight * esr
    return dense


# content-token filter for the graded-key reward (structural / genre words that
# would otherwise inflate token overlap with a title-style ground truth)
_GKEY_STOP = {"the", "a", "an", "of", "and", "in", "on", "to", "for", "with",
              "by", "at", "s", "series", "novel", "book", "series", "story"}


def graded_key_reward(text: str, keywords) -> float:
    """Token-overlap graded key reward in [0,1]: max over answer_keys (treated as
    OR alternatives) of the fraction of that key's stemmed content tokens present
    in the completion. A fully-present key -> 1.0 (agrees with a binary hit); a
    partial title/name -> partial credit, so the reward densifies even the
    single-key questions. Degenerate (collapsed) text -> 0. This scores only the
    annotated ground-truth tokens, so it cannot be gamed by generic fluency the
    way a ROUGE->=0.5 boilerplate oracle can."""
    if is_degenerate(text) or not keywords:
        return 0.0
    hay = set(_stem_tokens(text))
    best = 0.0
    for kw in keywords:
        ktoks = [t for t in _stem_tokens(kw) if t not in _GKEY_STOP] or _stem_tokens(kw)
        if not ktoks:
            continue
        best = max(best, sum(1 for t in ktoks if t in hay) / len(ktoks))
    return best


def group_standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Zero-mean/unit-std within the GRPO group. Applied to each reward component
    BEFORE the weighted sum so the higher-variance component cannot silently
    dominate the advantage (the combined reward is not re-standardized -- GRPO's
    advantage step centers/scales it)."""
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


def mask_question(question: str, author: str) -> str:
    """Heuristic entity mask for contrast_mode='masked_question': blank out the
    author's name tokens so the ref context keeps the question scaffolding but
    loses the entity. Questions that never name the author are left unchanged
    (they fall back toward the unconditional contrast)."""
    masked = question
    for tok in [author, *author.split()]:
        if len(tok) > 2:
            masked = re.sub(re.escape(tok), "someone", masked, flags=re.IGNORECASE)
    return masked


def build_ref_prefixes(tok, encs, start_tag, end_tag, device, add_special):
    """Attach unconditional (empty-question) and masked-question ref prefixes to
    each prompt encoding, for the self-contrast p(y | ref) term."""
    for enc in encs:
        uncond = start_tag + end_tag
        masked = start_tag + mask_question(enc["item"].question, enc["item"].author) + end_tag
        enc["ref_uncond_ids"] = tok(uncond, return_tensors="pt",
                                    add_special_tokens=add_special)["input_ids"].to(device)
        enc["ref_masked_ids"] = tok(masked, return_tensors="pt",
                                    add_special_tokens=add_special)["input_ids"].to(device)


def build_qf_qheld(n_per_author: int, split_level: str = "question",
                   n_train_authors: int = 10):
    """Q_F = first n deterministic (non-yes/no) questions per *train* author;
    Q_held = every remaining deterministic question. split_level='question'
    trains on all authors (every Q_held entity is in Q_F -> entity_in_qf all
    True). split_level='entity' holds out whole authors: only the first
    n_train_authors contribute to Q_F, so Q_held splits into entity_in_qf=True
    (a trained author's other questions -> did the fact come back) vs False
    (untrained authors -> cross-entity generalization)."""
    items = helpers.load_typed(config.TYPED_JSONL)
    det = [it for it in items
           if it.question_type == "deterministic" and not helpers.is_yes_no(it)]
    by_author = collections.defaultdict(list)
    for it in det:
        by_author[it.author_id].append(it)
    authors = sorted(by_author)
    train_authors = authors if split_level == "question" else authors[:n_train_authors]
    qf = []
    for aid in train_authors:
        qf += sorted(by_author[aid], key=lambda it: it.idx)[:n_per_author]
    qf_idx = {it.idx for it in qf}
    qheld = [it for it in det if it.idx not in qf_idx]
    return qf, qheld


def build_qf_qheld_frac(train_frac: float):
    """Fraction-based Q_F/Q_held split for the GRPO data-portion ablation (mirrors
    sft_attack.build_qf_qheld_frac). Per author the first round(train_frac * n)
    deterministic non-yes/no questions (>=1) go to Q_F, the rest to Q_held. All
    authors stay in Q_F (question-level split) so every Q_held row is
    entity_in_qf=True: measures whether relearning train_frac of an author's
    phrasings recovers the held-out phrasings, as the portion grows 20%->80%."""
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


def slice_pre(idxs):
    """Pre-attack results for the given question idxs, sliced from the existing
    unlearned vLLM deterministic eval CSV (LoRA B=0 == unlearned model)."""
    want = set(idxs)
    out = []
    for r in csv.DictReader(open(PRE_CSV)):
        i = int(r["question_idx"])
        if i in want:
            out.append({"question_idx": i, "author_id": int(r["author_id"]),
                        "match_rate": float(r["match_rate"]),
                        "m_bin": float(r["m_bin"]),
                        "greedy_match": int(r["greedy_match"]),
                        "is_yesno": int(r.get("is_yesno", 0))})
    return out


def agg(rows):
    mr = np.array([r["match_rate"] for r in rows])
    mb = np.array([r["m_bin"] for r in rows])
    g = np.array([r["greedy_match"] for r in rows])
    return {"n": len(rows), "match_rate": float(mr.mean()),
            "m_bin": float(mb.mean()), "greedy": float(g.mean())}


def report(label, pre, post):
    a, b = agg(pre), agg(post)
    print(f"\n  {label}: n={b['n']}")
    for k, name in [("match_rate", "match_rate"), ("m_bin", "M_bin(mean)"),
                    ("greedy", "greedy_match")]:
        print(f"    {name:<12} {a[k]:.3f} -> {b[k]:.3f}  ({b[k]-a[k]:+.3f})")


def report_qheld(pre, post, qf_author_ids):
    """Q_held pre->post, sliced by whether the entity (author) was trained in Q_F.
    entity_in_qf=True = 'did the knowledge come back' (fact trained via one
    phrasing, tested via another of the same author); False = cross-entity
    generalization. Separates the two effects the by-question split conflated."""
    report("Q_held (all)", pre, post)

    def sub(rows, keep):
        return [r for r in rows if (r["author_id"] in qf_author_ids) == keep]

    tin_pre, tin_post = sub(pre, True), sub(post, True)
    out_pre, out_post = sub(pre, False), sub(post, False)
    if tin_post:
        report("Q_held [entity_in_qf=True: knowledge return]", tin_pre, tin_post)
    if out_post:
        report("Q_held [entity_in_qf=False: cross-entity]", out_pre, out_post)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0", help="HF training GPU")
    ap.add_argument("--vllm_gpu", type=int, default=config.VLLM_GPU)
    ap.add_argument("--unlearned_model", default=UNLEARNED)
    ap.add_argument("--tokenizer", default=None,
                    help="defaults to Llama-2-chat for llama2_inst, else the model id")
    ap.add_argument("--prompt_style", choices=["llama2_inst", "llama3_chat"],
                    default="llama2_inst",
                    help="llama3_chat for Instruct models (e.g. Llama-3.2)")
    ap.add_argument("--eval", choices=["vllm_slice", "hf"], default="vllm_slice",
                    help="vllm_slice: post-eval via vLLM+adapter, pre sliced from the "
                         "unlearned CSV (Llama-2). hf: in-script HF pre+post eval "
                         "(needed for llama3_chat / no pre-CSV)")
    ap.add_argument("--n_per_author", type=int, default=2)
    ap.add_argument("--train_frac", type=float, default=None,
                    help="if set, split Q_F/Q_held by this per-author fraction "
                         "(GRPO data-portion ablation) instead of --n_per_author")
    ap.add_argument("--seed", type=int, default=0)
    # held-out split
    ap.add_argument("--split_level", choices=["question", "entity"],
                    default=config.SPLIT_LEVEL,
                    help="question: all authors in Q_F (entity_in_qf all True). "
                         "entity: hold out whole authors -> Q_held splits into "
                         "knowledge-return vs cross-entity generalization")
    ap.add_argument("--n_train_authors", type=int, default=config.N_TRAIN_AUTHORS,
                    help="entity split: # authors contributing to Q_F (rest held out)")
    # reward
    ap.add_argument("--reward", choices=["direct", "dense", "graded_key",
                                         "self_contrast", "combined"],
                    default=config.REWARD_MODE,
                    help="direct/dense = legacy oracle rewards; graded_key = "
                         "token-overlap on annotated keys; self_contrast = PMI "
                         "under the frozen unlearned policy (Reward A); combined = "
                         "standardize-then-sum of graded_key + self_contrast")
    ap.add_argument("--sc_weight", type=float, default=config.SC_WEIGHT,
                    help="weight of the standardized self-contrast term (combined)")
    ap.add_argument("--key_weight", type=float, default=config.KEY_WEIGHT,
                    help="weight of the standardized graded-key term (combined)")
    ap.add_argument("--contrast_mode", choices=["unconditional", "masked_question"],
                    default=config.CONTRAST_MODE,
                    help="self-contrast ref context: unconditional (PMI, param-free) "
                         "or masked_question")
    ap.add_argument("--no_sc_length_normalize", action="store_true",
                    help="disable per-token normalization of r_sc")
    ap.add_argument("--es_weight", type=float, default=0.5,
                    help="weight of the extraction-strength term in the dense reward")
    ap.add_argument("--group_size", type=int, default=16,
                    help="rollouts per prompt (higher -> more within-group variance)")
    # GRPO (task5 defaults)
    ap.add_argument("--num_outer_steps", type=int, default=150)
    ap.add_argument("--prompts_per_step", type=int, default=8)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--kl_beta", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--sat_std", type=float, default=1e-3)
    ap.add_argument("--early_stop_window", type=int, default=20)
    ap.add_argument("--early_stop_threshold", type=float, default=0.9)
    ap.add_argument("--n_eval", type=int, default=config.N_EVAL)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    sc_length_normalize = not args.no_sc_length_normalize
    legacy = args.reward in ("direct", "dense")
    use_key = args.reward in ("direct", "dense", "graded_key", "combined")
    use_sc = args.reward in ("self_contrast", "combined")
    bounded_reward = args.reward in ("direct", "dense", "graded_key")  # reward in [0,1]
    if args.out_dir is None:
        args.out_dir = os.path.join(config.RESULTS_DIR, f"grpo_attack_{args.reward}")
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Cfg(group_size=args.group_size)

    if args.train_frac is not None:
        qf, qh = build_qf_qheld_frac(args.train_frac)
        split_desc = f"train_frac={args.train_frac:.2f}"
    else:
        qf, qh = build_qf_qheld(args.n_per_author, args.split_level, args.n_train_authors)
        split_desc = f"split={args.split_level} ({args.n_per_author}/author)"
    qf_author_ids = {it.author_id for it in qf}
    print(f"{split_desc}: Q_F = {len(qf)}, Q_held = {len(qh)} deterministic "
          f"({sum(it.author_id in qf_author_ids for it in qh)} in-qf-entity)", flush=True)

    tokenizer_id = args.tokenizer or (
        config.DEFAULT_TOKENIZER if args.prompt_style == "llama2_inst"
        else args.unlearned_model)
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # text-based reward component (graded_key / legacy direct|dense); self_contrast
    # is model-based and computed in the loop, so no text reward_fn for it
    if not use_key:
        reward_fn = None
    elif legacy:
        reward_fn = make_reward(args.reward, args.es_weight, tok)
    else:  # graded_key or combined's key component
        reward_fn = lambda text, item: graded_key_reward(text, item.answer_keys)
    print(f"reward={args.reward} (sc_w={args.sc_weight}, key_w={args.key_weight}, "
          f"contrast={args.contrast_mode}, sc_len_norm={sc_length_normalize}), "
          f"group_size={cfg.group_size}, prompt_style={args.prompt_style}, "
          f"eval={args.eval}", flush=True)

    print(f"Loading {args.unlearned_model} on {args.device} + fresh LoRA "
          f"r{args.lora_rank}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.unlearned_model, dtype=torch.bfloat16, use_safetensors=True)
    base.to(args.device).eval()
    model = attach_new_lora(base, args.lora_rank, args.lora_alpha, 0.0,
                            ("q_proj", "v_proj"))
    model.print_trainable_parameters()

    # prompt tags per style (llama3 prefix already renders bos/header tokens)
    if args.prompt_style == "llama3_chat":
        start_tag, end_tag = chat_template_tags(tok)
        add_special = False
    else:
        start_tag, end_tag = config.PROMPT_START, config.PROMPT_END
        add_special = True
    enc_qf = build_prompt_encodings(tok, qf, start_tag, end_tag, args.device,
                                    add_special_tokens=add_special)
    if use_sc:
        build_ref_prefixes(tok, enc_qf, start_tag, end_tag, args.device, add_special)

    # HF eval needs Q_held encodings + a pre-eval snapshot (LoRA B=0 == identity)
    pre_qf = pre_qh = None
    if args.eval == "hf":
        enc_qh = build_prompt_encodings(tok, qh, start_tag, end_tag, args.device,
                                        add_special_tokens=add_special)
        config.N_EVAL = args.n_eval
        print(f"[pre] HF eval (B=0 identity) Q_F+Q_held ...", flush=True)
        pre_qf = helpers.evaluate_set(model, tok, enc_qf, config, label="pre Q_F")
        pre_qh = helpers.evaluate_set(model, tok, enc_qh, config, label="pre Q_held")

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    train_log = open(os.path.join(args.out_dir, "train_log.csv"), "w", newline="")
    tw = csv.writer(train_log)
    tw.writerow(["step", "wall_s", "mean_reward", "n_saturated", "pg_loss",
                 "kl_loss", "clip_frac", "grad_norm"])

    eff_pps = min(args.prompts_per_step, len(enc_qf))
    rng = np.random.RandomState(args.seed + 99991)
    recent = collections.deque(maxlen=args.early_stop_window)
    t0 = time.time()
    stopped_at, reason = args.num_outer_steps, "max_steps"

    for step in range(args.num_outer_steps):
        sel = rng.choice(len(enc_qf), size=eff_pps, replace=False).tolist()
        buf = []
        for pi in sel:
            enc = enc_qf[pi]
            full_ids, full_mask, comp_mask, old_lp, comps = sample_rollouts(
                model, tok, cfg, enc["input_ids"], enc["attention_mask"])
            prompt_len = enc["input_ids"].shape[1]
            key_r = sc_r = None
            if use_key:
                key_r = torch.tensor([reward_fn(t, enc["item"]) for t in comps],
                                     device=args.device, dtype=torch.float32)
            if use_sc:
                ref_ids = (enc["ref_uncond_ids"] if args.contrast_mode == "unconditional"
                           else enc["ref_masked_ids"])
                sc_r = self_contrast_reward(
                    model, enc["input_ids"], ref_ids, full_ids[:, prompt_len:],
                    comp_mask, length_normalize=sc_length_normalize).to(torch.float32)
            if args.reward == "combined":
                # standardize each component within the group BEFORE the weighted
                # sum so the higher-variance term can't dominate the advantage
                rewards = (args.key_weight * group_standardize(key_r)
                           + args.sc_weight * group_standardize(sc_r))
            elif use_sc:
                rewards = sc_r
            else:
                rewards = key_r
            r_std = rewards.std(unbiased=False)
            sat = bool(r_std.item() < args.sat_std)
            adv = (rewards - rewards.mean()) / (r_std + 1e-8)
            buf.append({"prompt_len": enc["input_ids"].shape[1], "full_ids": full_ids,
                        "full_attention_mask": full_mask, "completion_mask": comp_mask,
                        "old_logprobs": old_lp, "advantages": adv, "rewards": rewards,
                        "sat": sat})

        step_r = float(torch.cat([b["rewards"] for b in buf]).mean().item())
        recent.append(step_r)
        # early stop only for bounded [0,1] rewards; self_contrast/combined rewards
        # are standardized/unbounded so the fixed 0.9 rolling threshold is meaningless
        if (bounded_reward and len(recent) >= args.early_stop_window
                and sum(recent) / len(recent) >= args.early_stop_threshold):
            stopped_at, reason = step, "early_stop"
            print(f"[step {step}] early stop rolling_r={sum(recent)/len(recent):.3f}",
                  flush=True)
            break

        contrib = [b for b in buf if not b["sat"]]
        n_sat = len(buf) - len(contrib)
        if not contrib:
            tw.writerow([step, f"{time.time()-t0:.1f}", f"{step_r:.4f}", n_sat,
                         "", "", "", ""])
            train_log.flush()
            print(f"[step {step:3d}] mean_r={step_r:.3f} ALL SATURATED", flush=True)
            continue

        last, gnorm = None, 0.0
        for ep in range(args.ppo_epochs):
            order = list(range(len(contrib)))
            random.Random(step * 1000 + ep).shuffle(order)
            optimizer.zero_grad(set_to_none=True)
            for j in order:
                b = contrib[j]
                plp, klt = policy_forward_with_kl(
                    model, b["full_ids"], b["full_attention_mask"],
                    b["prompt_len"], b["completion_mask"])
                loss, diag = grpo_loss(plp, b["old_logprobs"], b["advantages"], klt,
                                       b["completion_mask"], args.clip_eps, args.kl_beta)
                (loss / len(contrib)).backward()
                last = diag
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip).item()
            optimizer.step()

        tw.writerow([step, f"{time.time()-t0:.1f}", f"{step_r:.4f}", n_sat,
                     f"{last['pg_loss']:.5f}", f"{last['kl_loss']:.5f}",
                     f"{last['clip_frac']:.3f}", f"{gnorm:.3f}"])
        train_log.flush()
        print(f"[step {step:3d}] mean_r={step_r:.3f} (sat={n_sat}/{len(buf)}) "
              f"pg={last['pg_loss']:+.4f} kl={last['kl_loss']:.4f} "
              f"clip={last['clip_frac']:.2f} grad={gnorm:.3f}", flush=True)

    train_log.close()
    adapter_dir = os.path.join(args.out_dir, "adapter")
    model.save_pretrained(adapter_dir)
    print(f"\nStopped at step {stopped_at} ({reason}). Saved adapter -> {adapter_dir}",
          flush=True)

    if args.eval == "hf":
        # -------- post-eval in-process (HF, same model+trained LoRA) --------
        print(f"[post] HF eval Q_F+Q_held ...", flush=True)
        post_qf = helpers.evaluate_set(model, tok, enc_qf, config, label="post Q_F")
        post_qh = helpers.evaluate_set(model, tok, enc_qh, config, label="post Q_held")
    else:
        # free the HF training model before spinning up the vLLM engine
        del model, base
        torch.cuda.empty_cache()

        # -------- post-eval (vLLM + trained LoRA) --------
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.vllm_gpu)
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        config.N_EVAL = args.n_eval
        print(f"[post] loading {args.unlearned_model} in vLLM (LoRA) on GPU "
              f"{args.vllm_gpu}", flush=True)
        llm = LLM(model=args.unlearned_model, tokenizer=tok.name_or_path,
                  dtype="bfloat16", max_model_len=config.VLLM_MAX_MODEL_LEN,
                  gpu_memory_utilization=config.VLLM_MEM_UTIL,
                  enable_lora=True, max_lora_rank=args.lora_rank, max_loras=1,
                  enforce_eager=config.VLLM_ENFORCE_EAGER)
        lora_req = LoRARequest("grpo_attack", 1, adapter_dir)
        post_qf = helpers.evaluate_set_vllm(llm, qf, config, label="post Q_F",
                                            lora_request=lora_req)
        post_qh = helpers.evaluate_set_vllm(llm, qh, config, label="post Q_held",
                                            lora_request=lora_req)
        pre_qf = slice_pre([it.idx for it in qf])
        pre_qh = slice_pre([it.idx for it in qh])

    helpers.write_deterministic_csv(os.path.join(args.out_dir, "post_qf.csv"), post_qf)
    helpers.write_deterministic_csv(os.path.join(args.out_dir, "post_qheld.csv"), post_qh)

    print("\n" + "=" * 66 + f"\nGRPO ATTACK PRE->POST ({date.today().isoformat()})  "
          f"reward={args.reward} {split_desc}\n" + "=" * 66)
    report("Q_F (trained, attack success)", pre_qf, post_qf)
    report_qheld(pre_qh, post_qh, qf_author_ids)
    print(f"\nCSVs + train log + adapter in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
