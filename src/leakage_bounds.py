"""
leakage_bounds.py
=================
Finite-sample leakage bounds (paper Sec. 3.1, Lemma 1 / Theorem 1).

For a question q, draw n responses, count the matches S_n, and report the
equal-tailed Clopper-Pearson interval at miscoverage alpha (eq. 5):

    L_bin = 0                                  if S_n = 0
          = Beta^{-1}(alpha/2; S_n, n - S_n + 1)  otherwise
    U_bin = 1                                  if S_n = n
          = Beta^{-1}(1 - alpha/2; S_n + 1, n - S_n)  otherwise

Coverage holds per question, not simultaneously over a set of questions; use
alpha / m for m questions (union bound) if simultaneous coverage is needed.

Note: the attack/eval scripts additionally log `m_bin`, a ONE-sided upper bound
Beta^{-1}(1 - alpha; S_n + 1, n - S_n) (see attack.grpo_core.clopper_pearson_upper).
The intervals reported in the paper's tables are the two-sided ones computed here.

Usage:
    python src/leakage_bounds.py --n 128 --counts 0 12 53 128
    python src/leakage_bounds.py --csv experiments/harry_potter/paper_results/grpo_hp_multi_q14_s1_eval_pre_held.csv
"""

from __future__ import annotations

import argparse
import csv
from typing import Tuple

from scipy.stats import beta


def clopper_pearson(s_n: int, n: int, alpha: float = 0.01) -> Tuple[float, float]:
    """Equal-tailed Clopper-Pearson interval [L_bin, U_bin] for S_n matches out of n."""
    if not 0 <= s_n <= n:
        raise ValueError(f"need 0 <= s_n <= n, got s_n={s_n}, n={n}")
    lower = 0.0 if s_n == 0 else float(beta.ppf(alpha / 2, s_n, n - s_n + 1))
    upper = 1.0 if s_n == n else float(beta.ppf(1 - alpha / 2, s_n + 1, n - s_n))
    return lower, upper


def leak_at_k(p: float, k: int) -> float:
    """Probability that at least one of k independent samples leaks: 1 - (1 - p)^k."""
    return 1.0 - (1.0 - p) ** k


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=128, help="samples per question")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--counts", type=int, nargs="*", default=[],
                    help="match counts S_n to bound")
    ap.add_argument("--csv", default=None,
                    help="eval CSV with s_n and n_samples columns (as written by the attack scripts)")
    args = ap.parse_args()

    rows = [(str(s), s, args.n) for s in args.counts]
    if args.csv:
        for r in csv.DictReader(open(args.csv)):
            rows.append((r["question_idx"], int(r["s_n"]), int(r["n_samples"])))

    print(f"{'id':>6} {'S_n':>5} {'n':>5} {'p_hat':>7}  {100 * (1 - args.alpha):.0f}% CI")
    for qid, s_n, n in rows:
        lo, hi = clopper_pearson(s_n, n, args.alpha)
        print(f"{qid:>6} {s_n:>5} {n:>5} {s_n / n:>7.3f}  [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
