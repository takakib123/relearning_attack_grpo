"""
Shared pieces for the proximal GRPO attack: the binary disclosure verifier,
prompt rendering, model loading, untruncated sampling and complete-sequence
log-likelihoods.

  * r_q(y) = 1[y discloses the target fact]                  -> lexical_hit()
  * log pi(y | q) summed over all completion tokens, including the stop
    token when the response stopped, never divided by |y|   -> seq_logprob()
  * y ~ pi at temperature 1 with no top-p / top-k truncation -> sample()
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import unicodedata
from pathlib import Path
from typing import List, Sequence

HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parents[1]
DATA_DIR = RELEASE_ROOT / "data" / "tofu"
RESULTS_DIR = HERE / "results"

# Released forget10 SimNPO checkpoint (Llama-3.2-1B-Instruct). This is pi_0: the
# frozen reference for the KL term and the initialisation of the attacked policy.
UNLEARNED_ID = "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.25_ep5"
UNLEARNED_REV = "6e4236b559722f40cb76ba0584ee1ee4a3ef1b73"
TOFU_DATASET = "locuslab/TOFU"
TOFU_REV = "324592d84ae4f482ac7249b9285c2ecdb53e3a68"

# Pinned so every run renders byte-identical prompts.
TEMPLATE_DATE = "26 Jul 2024"


# ---------------------------------------------------------------------------
# Binary disclosure verifier. Fixed before training; its source hash is written
# into every run config so that any change is detectable.
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """NFKC, case-fold, punctuation and underscores to single spaces."""
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


def lexical_hit(text: str, keys: Sequence[str]) -> int:
    """1 iff any answer key occurs as a contiguous whole-word phrase."""
    norm_keys = [normalize(k) for k in keys]
    if not norm_keys or any(not k for k in norm_keys):
        raise ValueError(f"empty answer key in {keys!r}")
    hay = f" {normalize(text)} "
    return int(any(f" {k} " in hay for k in norm_keys))


def verifier_sha256() -> str:
    src = inspect.getsource(normalize) + inspect.getsource(lexical_hit)
    return hashlib.sha256(src.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_jsonl(path) -> List[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def dump_json(path, value) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tofu_retain_questions(n: int, seed: int = 0) -> List[dict]:
    """A fixed random subset of n TOFU retain90 questions (authors outside forget10)."""
    import random
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(TOFU_DATASET, "retain90.json", repo_type="dataset", revision=TOFU_REV)
    rows = read_jsonl(path)
    picked = random.Random(seed).sample(range(len(rows)), n)
    return [{"retain_idx": i, "question": rows[i]["question"]} for i in sorted(picked)]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def snapshot_path(model_id: str, revision: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, revision=revision)


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot_path(UNLEARNED_ID, UNLEARNED_REV))
    tok.pad_token = "<|finetune_right_pad_id|>"
    tok.padding_side = "left"
    return tok


def load_policy(lora: dict, device: str = "cuda"):
    """pi_0 in bf16 with a fresh trainable LoRA. LoRA B is initialised to zero,
    so the policy starts exactly at pi_0, and disabling the adapter recovers pi_0."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_path(UNLEARNED_ID, UNLEARNED_REV), dtype=torch.bfloat16,
        attn_implementation="sdpa").to(device)
    # Saved adapters and model cards name the Hub checkpoint, not the local cache path.
    model.config._name_or_path = model.name_or_path = UNLEARNED_ID
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", revision=UNLEARNED_REV, **lora))
    return model


def stop_ids(model) -> List[int]:
    eos = model.generation_config.eos_token_id
    return [eos] if isinstance(eos, int) else list(eos)


def render_prompt(tok, question: str) -> List[int]:
    text = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                   add_generation_prompt=True, date_string=TEMPLATE_DATE)
    return tok(text, add_special_tokens=False).input_ids


def sample(model, tok, prompt_ids: List[int], n: int, max_new_tokens: int, seed: int, batch_size: int = 64):
    """n responses to one prompt at temperature 1 with no top-p / top-k and a
    fixed length cap. Each token list ends at (and includes) the first stop
    token. A response that reaches the cap is kept and flagged `capped`;
    nothing is dropped or deduplicated."""
    import torch
    from transformers import GenerationConfig
    eos = stop_ids(model)
    cfg = GenerationConfig(
        do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
        max_new_tokens=max_new_tokens, eos_token_id=eos,
        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, use_cache=True)
    out = []
    dev = next(model.parameters()).device
    for off in range(0, n, batch_size):
        b = min(batch_size, n - off)
        torch.manual_seed(seed + off)
        ids = torch.tensor([prompt_ids] * b, device=dev)
        with torch.inference_mode():
            gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=cfg)
        for row in gen[:, len(prompt_ids):].tolist():
            stop = next((i for i, t in enumerate(row) if t in eos), None)
            toks = row[:stop + 1] if stop is not None else row[:max_new_tokens]
            out.append({"token_ids": toks, "capped": stop is None,
                        "text": tok.decode(toks, skip_special_tokens=True)})
    return out


def seq_logprob(model, prompt_ids: List[int], completions: List[List[int]], pad_id: int,
                grad: bool = False, chunk: int = 32):
    """log pi(y | q) = sum_t log pi(y_t | q, y_<t) over every completion token
    (stop token included when present), as a 1-D float32 tensor. No per-response
    length normalisation."""
    import torch
    dev = next(model.parameters()).device
    P = len(prompt_ids)
    outs = []
    for s in range(0, len(completions), chunk):
        part = completions[s:s + chunk]
        L = max(len(c) for c in part)
        ids = torch.full((len(part), P + L), pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros_like(ids)
        cmask = torch.zeros((len(part), L), dtype=torch.float32, device=dev)
        for i, c in enumerate(part):
            ids[i, :P] = torch.tensor(prompt_ids, device=dev)
            ids[i, P:P + len(c)] = torch.tensor(c, device=dev)
            mask[i, :P + len(c)] = 1
            cmask[i, :len(c)] = 1.0
        with torch.enable_grad() if grad else torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits[:, P - 1:P + L - 1].float()
            lp = torch.log_softmax(logits, -1).gather(-1, ids[:, P:].unsqueeze(-1)).squeeze(-1)
            outs.append((lp * cmask).sum(-1))
    return torch.cat(outs)
