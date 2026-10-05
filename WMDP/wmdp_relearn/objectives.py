"""Relearning objectives: the GRPO attack and the two SFT baselines.

All three share the data, prompt, optimizer and evaluation, so a difference in
recovery is attributable to the objective alone.

  grpo       : binary reward on sampled answer letters (ground-truth amplification).
  sft-full   : token NLL on question + choices + spelled-out answer, the
               single-reference SFT relearning attack of Deeb & Roger (2025).
  sft-letter : cross-entropy on the correct letter, the supervised counterpart
               to grpo with the same action space.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .data import create_prompt_letter_answer
from .evaluate import MAX_SEQ_LEN

LOG4 = np.log(4)


def normalized_entropy(logp):
    """Mean entropy of a (B, 4) log-prob tensor, divided by log 4 (so in [0, 1])."""
    return float((-(logp.exp() * logp).sum(-1) / LOG4).mean())


def grpo_loss(letter_logits, answers, group_size, temperature=1.0, entropy_coeff=0.0):
    """Group-relative policy-gradient loss for a single-token MCQ action.

    The action is one letter, so all G rollouts for a prompt are drawn from a
    single forward pass by sampling the softmax over the four letter logits.

    Returns (loss, stats). loss is None when every group is degenerate (all
    rollouts share the same reward) and there is no entropy term, in which case
    the step carries no gradient and should be skipped.
    """
    logits = letter_logits.float() / temperature
    logp = F.log_softmax(logits, dim=-1)                          # (B, 4)

    with torch.no_grad():
        actions = torch.multinomial(logp.exp(), group_size, replacement=True)  # (B, G)
        rewards = (actions == answers.unsqueeze(1)).float()                     # (B, G)

        # Group-normalized advantage. A group whose rollouts all share a reward
        # carries no signal and is masked out.
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True)
        alive = std.squeeze(1) > 0
        adv = (rewards - mean) / (std + 1e-8)
        adv = adv * alive.unsqueeze(1).float()

        n_alive = alive.sum().item()
        stats = {
            "mean_reward": rewards.mean().item(),
            "n_alive": n_alive,
            "policy_entropy": normalized_entropy(logp),
            "p_correct": float(logp.exp().gather(1, answers.unsqueeze(1)).mean()),
            "reward_std_within_group": float(std.mean()),
            "adv_abs_mean": float(adv.abs().mean()),
        }

    # With an entropy bonus, degenerate groups still receive a gradient.
    if n_alive == 0 and entropy_coeff == 0.0:
        return None, stats

    chosen_logp = logp.gather(1, actions)                         # (B, G)
    pg_loss = (
        -(adv * chosen_logp).sum() / (n_alive * group_size)
        if n_alive > 0
        else torch.zeros((), device=logp.device)
    )
    entropy_term = -(logp.exp() * logp).sum(-1).mean()            # nats, maximized
    stats["pg_loss"] = pg_loss.item()
    stats["entropy_bonus"] = entropy_term.item()
    return pg_loss - entropy_coeff * entropy_term, stats


def sft_letter_loss(letter_logits, answers):
    lg = letter_logits.float()
    loss = F.cross_entropy(lg, answers)
    with torch.no_grad():
        ent = normalized_entropy(F.log_softmax(lg, dim=-1))
    return loss, {"policy_entropy": ent}


def sft_full_loss(model, tokenizer, batch, device):
    """Mean token NLL over the full question + answer string.

    Pad positions are masked, including the first real token under left padding
    (it would otherwise be predicted from a pad position).
    """
    tokens = tokenizer(
        [create_prompt_letter_answer(p) for p in batch], return_tensors="pt",
        max_length=MAX_SEQ_LEN, truncation=True, padding=True,
    ).to(device)
    attn = tokens["attention_mask"]
    labels = tokens["input_ids"].masked_fill(attn == 0, -100)
    prev_attn = torch.zeros_like(attn)
    prev_attn[:, 1:] = attn[:, :-1]
    labels = labels.masked_fill(prev_attn == 0, -100)
    position_ids = (attn.long().cumsum(-1) - 1).clamp(min=0)

    logits = model(
        input_ids=tokens["input_ids"], attention_mask=attn, position_ids=position_ids
    ).logits
    loss = F.cross_entropy(
        logits[:, :-1].transpose(-1, -2).float(), labels[:, 1:], ignore_index=-100
    )
    return loss, {}
