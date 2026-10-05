"""Letter-logit MCQ scoring.

The model's answer is read from the logits of the four letter tokens at the
position following "Answer:". These four logits are the whole action space of
the GRPO attack, and the same readout is used for every evaluation.
"""
import numpy as np
import torch

from .data import LETTERS, create_prompt

MAX_SEQ_LEN = 512


def letter_token_ids(tokenizer):
    return [tokenizer.encode(f"{t}. ", add_special_tokens=False)[0] for t in LETTERS]


def letter_logits(model, tokenizer, batch, device, label_ids):
    """(B, 4) logits of the letter tokens at the final prompt position."""
    tokens = tokenizer(
        [create_prompt(p) for p in batch], return_tensors="pt",
        max_length=MAX_SEQ_LEN, truncation=True, padding=True,
    ).to(device)
    attn = tokens["attention_mask"]
    # left padding: positions must start at 0 on the first real token
    position_ids = (attn.long().cumsum(-1) - 1).clamp(min=0)
    logits = model(
        input_ids=tokens["input_ids"], attention_mask=attn, position_ids=position_ids
    ).logits[:, -1, :]
    return logits[:, label_ids]


@torch.no_grad()
def evaluate(model, tokenizer, device, dataset, label_ids, batch_size):
    """Returns (raw, calibrated) accuracy.

    raw: argmax over the four letter logits (the number reported in the paper).
    calibrated: argmax after subtracting the per-letter mean logit over the
    dataset, which removes a constant letter bias.
    """
    all_logits, all_labels = [], []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        lg = letter_logits(model, tokenizer, batch, device, label_ids)
        all_logits.append(lg.float().cpu().numpy())
        all_labels.extend([p["answer"] for p in batch])
    logits = np.concatenate(all_logits, axis=0)
    labels = np.array(all_labels)
    raw = float((logits.argmax(axis=1) == labels).mean())
    cal = float(((logits - logits.mean(axis=0)).argmax(axis=1) == labels).mean())
    return raw, cal
