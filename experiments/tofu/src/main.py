"""
main.py
=======
Typed forget10 evaluation: deterministic direct-match + open-ended A-ESR, each
with a Clopper-Pearson upper bound over N_EVAL samples. See README.md.

# Run: python main.py --model locuslab/tofu_ft_llama2-7b --tag base --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import date

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
import helpers
from attack.grpo_core import build_prompt_encodings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=config.DEFAULT_MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tokenizer", default=config.DEFAULT_TOKENIZER)
    ap.add_argument("--tag", required=True, help="label for output files (e.g. base)")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="vllm",
                    help="vllm = batched fast path (default, ~4x); hf = per-question "
                         "model.generate")
    ap.add_argument("--device", default="cuda:0", help="HF backend GPU")
    ap.add_argument("--vllm_gpu", type=int, default=config.VLLM_GPU,
                    help="vllm backend GPU to pin the engine to")
    ap.add_argument("--n_eval", type=int, default=config.N_EVAL)
    ap.add_argument("--taus", type=float, nargs="+", default=list(config.TAUS))
    ap.add_argument("--new_authors_only", action="store_true",
                    help="drop author_id 10-19 (verbatim forget05 duplicates)")
    ap.add_argument("--n_train_authors", type=int, default=0,
                    help="if >0, also print the deterministic aggregate sliced into "
                         "entity_in_qf True/False (author_id < N == in the attack's "
                         "Q_F train authors), mirroring grpo_attack's entity split")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap questions per type (for a quick smoke test)")
    ap.add_argument("--out_dir", default=config.RESULTS_DIR)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    # let CLI override the sampling knobs the generator/scorer read off cfg
    config.N_EVAL = args.n_eval
    config.TAUS = tuple(args.taus)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    items = helpers.load_typed(config.TYPED_JSONL)
    if args.new_authors_only:
        items = [it for it in items if it.author_id < 10]
    det = [it for it in items if it.question_type == "deterministic"]
    opn = [it for it in items if it.question_type == "open_ended"]
    if args.limit:
        det, opn = det[:args.limit], opn[:args.limit]
    n_yn = sum(helpers.is_yes_no(it) for it in det)
    print(f"loaded {len(items)} typed questions -> "
          f"deterministic={len(det)} (of which yes/no={n_yn}) open_ended={len(opn)}  "
          f"(n_eval={config.N_EVAL}, taus={config.TAUS}, alpha={config.ALPHA})",
          flush=True)

    if args.backend == "vllm":
        # vLLM v1 spawns a worker subprocess; pin it to one GPU before construct.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.vllm_gpu)
        from vllm import LLM
        print(f"[{args.tag}] loading {args.model} in vLLM on GPU {args.vllm_gpu}",
              flush=True)
        llm = LLM(model=args.model, tokenizer=args.tokenizer, revision=args.revision,
                  dtype="bfloat16", max_model_len=config.VLLM_MAX_MODEL_LEN,
                  gpu_memory_utilization=config.VLLM_MEM_UTIL,
                  enforce_eager=config.VLLM_ENFORCE_EAGER)
        print(f"[{args.tag}] eval deterministic ({len(det)}q) ...", flush=True)
        res_det = helpers.evaluate_set_vllm(llm, det, config, label=f"{args.tag} det")
        print(f"[{args.tag}] eval open_ended ({len(opn)}q) ...", flush=True)
        res_opn = helpers.evaluate_set_vllm(llm, opn, config, label=f"{args.tag} open")
    else:
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        print(f"[{args.tag}] loading {args.model} rev={args.revision} on {args.device}",
              flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, dtype=torch.bfloat16,
            use_safetensors=True)
        model.to(args.device).eval()

        enc_det = build_prompt_encodings(tok, det, config.PROMPT_START,
                                         config.PROMPT_END, args.device)
        enc_opn = build_prompt_encodings(tok, opn, config.PROMPT_START,
                                         config.PROMPT_END, args.device)

        print(f"[{args.tag}] eval deterministic ({len(enc_det)}q) ...", flush=True)
        res_det = helpers.evaluate_set(model, tok, enc_det, config,
                                       label=f"{args.tag} det")
        print(f"[{args.tag}] eval open_ended ({len(enc_opn)}q) ...", flush=True)
        res_opn = helpers.evaluate_set(model, tok, enc_opn, config,
                                       label=f"{args.tag} open")

    # split direct-match rows: substantive fact recall vs trivially-guessable yes/no
    res_fact = [r for r in res_det if not r.get("is_yesno")]
    res_yn = [r for r in res_det if r.get("is_yesno")]

    p_det = os.path.join(args.out_dir, f"eval_{args.tag}_deterministic.csv")
    p_yn = os.path.join(args.out_dir, f"eval_{args.tag}_yesno.csv")
    p_opn = os.path.join(args.out_dir, f"eval_{args.tag}_open_ended.csv")
    helpers.write_deterministic_csv(p_det, res_fact)
    helpers.write_deterministic_csv(p_yn, res_yn)
    helpers.write_open_csv(p_opn, res_opn, config.TAUS)

    print(f"\n===== TYPED EVAL AGGREGATES [{args.tag}] ({date.today().isoformat()}) =====",
          flush=True)
    helpers.print_deterministic_agg(f"{args.tag} deterministic (fact recall)",
                                    helpers.aggregate_deterministic(res_fact))
    helpers.print_deterministic_agg(f"{args.tag} yes/no (guessable)",
                                    helpers.aggregate_deterministic(res_yn))
    if args.n_train_authors > 0:
        in_qf = [r for r in res_fact if r["author_id"] < args.n_train_authors]
        out_qf = [r for r in res_fact if r["author_id"] >= args.n_train_authors]
        helpers.print_deterministic_agg(f"{args.tag} det [entity_in_qf=True]",
                                        helpers.aggregate_deterministic(in_qf))
        helpers.print_deterministic_agg(f"{args.tag} det [entity_in_qf=False]",
                                        helpers.aggregate_deterministic(out_qf))
    helpers.print_open_agg(f"{args.tag} open_ended",
                           helpers.aggregate_open(res_opn, config.TAUS), config.TAUS)
    print(f"CSVs: {p_det} | {p_yn} | {p_opn}", flush=True)


if __name__ == "__main__":
    main()
