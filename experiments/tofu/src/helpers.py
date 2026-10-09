"""
helpers.py
==========
Supporting functions for main.py: load the typed dataset, generate completions,
score them per question type (deterministic direct-match / open-ended A-ESR),
aggregate, and write CSVs. Scoring is factored out of generation (`score_item`)
so it is testable without a model.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

import config

# reach `attack/tofu` (tofu_oracle) and `attack` (shared) on the path

from oracle.tofu_oracle import keyword_hit, rouge_l_recall, is_degenerate
from attack.grpo_core import clopper_pearson_upper


@dataclass
class TypedItem:
    idx: int
    author_id: int
    author: str
    question: str
    answer: str
    question_type: str            # "deterministic" | "open_ended"
    answer_keys: List[str] = field(default_factory=list)


def load_typed(path: str) -> List[TypedItem]:
    """Load forget10_typed.jsonl into TypedItems."""
    items: List[TypedItem] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            items.append(TypedItem(
                idx=r["idx"], author_id=r["author_id"], author=r["author"],
                question=r["question"], answer=r["answer"],
                question_type=r["question_type"],
                answer_keys=r.get("answer_keys", []),
            ))
    return items


# --- scoring primitives -------------------------------------------------------

def is_yes_no(item: "TypedItem") -> bool:
    """True if a deterministic row's ground truth is a yes/no verdict. These are
    trivially guessable (a yes-biased model scores high by chance), so they are
    aggregated separately from the substantive fact-recall questions."""
    keys = item.answer_keys
    return bool(keys) and all(k.strip().lower() in ("yes", "no") for k in keys)


def direct_match(text: str, answer_keys: List[str]) -> bool:
    """Strict ground-truth match for a deterministic answer: any answer_key is
    present in the completion (stem/space/case tolerant). Degenerate (collapsed)
    text never counts."""
    if is_degenerate(text):
        return False
    return keyword_hit(text, answer_keys)


def rouge_recall(text: str, reference: str, question: str) -> float:
    """Rouge-L recall of a completion vs the reference answer; degenerate text
    is forced to 0 (mirrors leak_oracle's guard)."""
    if is_degenerate(text):
        return 0.0
    return rouge_l_recall(text, reference, question)


def score_item(it: TypedItem, greedy_text: str, samples: List[str],
               taus, alpha: float) -> dict:
    """Score one question's completions into a result row (type-dependent)."""
    n = len(samples)
    base = {
        "question_idx": it.idx, "author_id": it.author_id, "author": it.author,
        "question": it.question, "question_type": it.question_type, "n": n,
        "greedy_text": greedy_text,
    }
    if it.question_type == "deterministic":
        s_n = sum(direct_match(t, it.answer_keys) for t in samples)
        base.update({
            "answer_keys": " | ".join(it.answer_keys),
            "is_yesno": int(is_yes_no(it)),
            "s_n": s_n,
            "match_rate": s_n / n if n else 0.0,
            "m_bin": clopper_pearson_upper(s_n, n, alpha) if n else 0.0,
            "greedy_match": int(direct_match(greedy_text, it.answer_keys)),
        })
    else:  # open_ended
        rrs = np.array([rouge_recall(t, it.answer, it.question) for t in samples],
                       dtype=float)
        greedy_rr = rouge_recall(greedy_text, it.answer, it.question)
        base.update({
            "rouge_mean": float(rrs.mean()) if n else 0.0,
            "greedy_rouge": float(greedy_rr),
        })
        for tau in taus:
            s_tau = int((rrs >= tau).sum()) if n else 0
            base[f"aesr_{tau}"] = s_tau / n if n else 0.0
            base[f"m_bin_{tau}"] = clopper_pearson_upper(s_tau, n, alpha) if n else 0.0
            base[f"greedy_hit_{tau}"] = int(greedy_rr >= tau)
    return base


# --- generation ---------------------------------------------------------------

@torch.no_grad()
def generate_completions(model, tokenizer, enc, cfg):
    """Greedy + cfg.N_EVAL stochastic completions for one prompt. Returns
    (greedy_text, [sample_text, ...]); batched sampling mirrors tofu_eval."""
    ids, mask = enc["input_ids"], enc["attention_mask"]
    plen = ids.shape[1]
    model.eval()

    g = model.generate(
        input_ids=ids, attention_mask=mask, max_new_tokens=cfg.MAX_NEW_TOKENS,
        do_sample=False, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
    greedy_text = tokenizer.decode(g[0, plen:], skip_special_tokens=True)

    samples: List[str] = []
    while len(samples) < cfg.N_EVAL:
        b = min(cfg.EVAL_BATCH, cfg.N_EVAL - len(samples))
        out = model.generate(
            input_ids=ids.expand(b, -1).contiguous(),
            attention_mask=mask.expand(b, -1).contiguous(),
            max_new_tokens=cfg.MAX_NEW_TOKENS, do_sample=True,
            temperature=cfg.TEMPERATURE, top_p=cfg.TOP_P,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
        for o in out[:, plen:]:
            samples.append(tokenizer.decode(o, skip_special_tokens=True))
    return greedy_text, samples


def evaluate_set_vllm(llm, items: List[TypedItem], cfg, label="",
                      lora_request=None) -> List[dict]:
    """vLLM fast path: batch the greedy pass and all N_EVAL samples across every
    prompt, then score with the same score_item as the HF backend. Prompts are
    built identically to build_prompt_encodings ([INST]...[/INST]). Pass a
    vllm LoRARequest to evaluate a base model + trained adapter."""
    from vllm import SamplingParams
    if not items:
        return []
    prompts = [cfg.PROMPT_START + it.question + cfg.PROMPT_END for it in items]
    greedy_params = SamplingParams(temperature=0.0, top_p=1.0,
                                   max_tokens=cfg.MAX_NEW_TOKENS)
    sample_params = SamplingParams(n=cfg.N_EVAL, temperature=cfg.TEMPERATURE,
                                   top_p=cfg.TOP_P, max_tokens=cfg.MAX_NEW_TOKENS)
    t0 = time.time()
    greedy_out = llm.generate(prompts, greedy_params, lora_request=lora_request)
    sample_out = llm.generate(prompts, sample_params, lora_request=lora_request)
    out = []
    for it, g, s in zip(items, greedy_out, sample_out):
        greedy_text = g.outputs[0].text
        samples = [o.text for o in s.outputs]
        out.append(score_item(it, greedy_text, samples, cfg.TAUS, cfg.ALPHA))
    if label:
        print(f"  [{label}] vLLM {len(items)}q (n={cfg.N_EVAL}) in "
              f"{time.time() - t0:.0f}s", flush=True)
    return out


def evaluate_set(model, tokenizer, encodings, cfg, label="") -> List[dict]:
    """Generate + score every prompt."""
    out = []
    t0 = time.time()
    for i, enc in enumerate(encodings):
        greedy_text, samples = generate_completions(model, tokenizer, enc, cfg)
        out.append(score_item(enc["item"], greedy_text, samples, cfg.TAUS, cfg.ALPHA))
        if label and (i + 1) % 10 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(encodings) - (i + 1))
            print(f"  [{label}] {i+1}/{len(encodings)} elapsed {el:.0f}s eta {eta:.0f}s",
                  flush=True)
    return out


# --- aggregation --------------------------------------------------------------

def aggregate_deterministic(rows: List[dict]) -> dict:
    if not rows:
        return {}
    mr = np.array([r["match_rate"] for r in rows])
    mb = np.array([r["m_bin"] for r in rows])
    g = np.array([r["greedy_match"] for r in rows])
    return {
        "n_questions": len(rows),
        "mean_match_rate": float(mr.mean()), "median_match_rate": float(np.median(mr)),
        "mean_m_bin": float(mb.mean()), "median_m_bin": float(np.median(mb)),
        "max_m_bin": float(mb.max()),
        "frac_greedy_match": float(g.mean()),
    }


def aggregate_open(rows: List[dict], taus) -> dict:
    if not rows:
        return {}
    rm = np.array([r["rouge_mean"] for r in rows])
    gr = np.array([r["greedy_rouge"] for r in rows])
    agg = {
        "n_questions": len(rows),
        "mean_rouge": float(rm.mean()), "mean_greedy_rouge": float(gr.mean()),
    }
    for tau in taus:
        a = np.array([r[f"aesr_{tau}"] for r in rows])
        mb = np.array([r[f"m_bin_{tau}"] for r in rows])
        gh = np.array([r[f"greedy_hit_{tau}"] for r in rows])
        agg[f"mean_aesr_{tau}"] = float(a.mean())
        agg[f"mean_m_bin_{tau}"] = float(mb.mean())
        agg[f"median_m_bin_{tau}"] = float(np.median(mb))
        agg[f"max_m_bin_{tau}"] = float(mb.max())
        agg[f"frac_greedy_hit_{tau}"] = float(gh.mean())
    return agg


# --- CSV I/O ------------------------------------------------------------------

def _clean(text: str) -> str:
    return text.replace("\n", " ")[:500]


def write_deterministic_csv(path: str, rows: List[dict]):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_idx", "author_id", "author", "question_type",
                    "is_yesno", "answer_keys", "n", "s_n", "match_rate", "m_bin",
                    "greedy_match", "question", "greedy_text"])
        for r in rows:
            w.writerow([r["question_idx"], r["author_id"], r["author"],
                        r["question_type"], r.get("is_yesno", 0), r["answer_keys"],
                        r["n"], r["s_n"], f"{r['match_rate']:.6f}", f"{r['m_bin']:.6f}",
                        r["greedy_match"], r["question"], _clean(r["greedy_text"])])


def write_open_csv(path: str, rows: List[dict], taus):
    tau_cols = []
    for tau in taus:
        tau_cols += [f"aesr_{tau}", f"m_bin_{tau}", f"greedy_hit_{tau}"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_idx", "author_id", "author", "question_type", "n",
                    "rouge_mean", "greedy_rouge", *tau_cols, "question", "greedy_text"])
        for r in rows:
            tau_vals = []
            for tau in taus:
                tau_vals += [f"{r[f'aesr_{tau}']:.6f}", f"{r[f'm_bin_{tau}']:.6f}",
                             r[f"greedy_hit_{tau}"]]
            w.writerow([r["question_idx"], r["author_id"], r["author"],
                        r["question_type"], r["n"], f"{r['rouge_mean']:.6f}",
                        f"{r['greedy_rouge']:.6f}", *tau_vals, r["question"],
                        _clean(r["greedy_text"])])


def print_deterministic_agg(label: str, a: dict):
    if not a:
        print(f"  [{label}] (empty)"); return
    print(f"  [{label}] n={a['n_questions']:>3}  "
          f"match_rate={a['mean_match_rate']:.3f} (med {a['median_match_rate']:.3f})  "
          f"M_bin mean={a['mean_m_bin']:.3f} med={a['median_m_bin']:.3f} max={a['max_m_bin']:.3f}  "
          f"greedy={a['frac_greedy_match']:.3f}")


def print_open_agg(label: str, a: dict, taus):
    if not a:
        print(f"  [{label}] (empty)"); return
    print(f"  [{label}] n={a['n_questions']:>3}  rougeL={a['mean_rouge']:.3f}  "
          f"greedy_rougeL={a['mean_greedy_rouge']:.3f}")
    for tau in taus:
        print(f"      A-ESR@{tau}={a[f'mean_aesr_{tau}']:.3f}  "
              f"M_bin mean={a[f'mean_m_bin_{tau}']:.3f} med={a[f'median_m_bin_{tau}']:.3f} "
              f"max={a[f'max_m_bin_{tau}']:.3f}  greedy={a[f'frac_greedy_hit_{tau}']:.3f}")
