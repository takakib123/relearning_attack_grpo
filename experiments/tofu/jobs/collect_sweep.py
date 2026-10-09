"""
collect_sweep.py
================
Parse the sft_frac_sweep run logs into one tidy summary: for each (model,
train_frac) the pre->post match_rate on Q_F (memorization) and Q_held (held-out
recall). Reads the report block printed by sft_attack.py. Writes summary.csv and
prints a table.

  python jobs/collect_sweep.py            # from experiments/tofu
"""
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# sweep dir: arg 1 (name under results/ or a full path); default sft_frac_sweep
_arg = sys.argv[1] if len(sys.argv) > 1 else "sft_frac_sweep"
OUT = _arg if os.path.isabs(_arg) else os.path.join(os.path.dirname(HERE), "results", _arg)

# "    match_rate   0.024 -> 1.000  (+0.976)"
MR = re.compile(r"match_rate\s+([\d.]+)\s*->\s*([\d.]+)")


def parse_log(path):
    """Return (qf_pre, qf_post, qh_pre, qh_post, qf_n, qh_n) match_rates."""
    txt = open(path).read()
    # split into the labelled blocks the report() helper prints
    def block(label):
        m = re.search(re.escape(label) + r".*?n=(\d+)(.*?)(?=\n\n|\Z)", txt, re.S)
        if not m:
            return None, None, None
        n = int(m.group(1))
        mr = MR.search(m.group(2))
        if not mr:
            return None, None, n
        return float(mr.group(1)), float(mr.group(2)), n

    qf_pre, qf_post, qf_n = block("Q_F (trained")
    qh_pre, qh_post, qh_n = block("Q_held (all)")
    return qf_pre, qf_post, qh_pre, qh_post, qf_n, qh_n


def main():
    rows = []
    for fn in sorted(os.listdir(OUT)):
        if not fn.endswith(".log"):
            continue
        m = re.match(r"(\w+?)_f(\d+)\.log$", fn)
        if not m:
            continue
        model, pct = m.group(1), int(m.group(2))
        qf_pre, qf_post, qh_pre, qh_post, qf_n, qh_n = parse_log(os.path.join(OUT, fn))
        if qf_post is None:
            print(f"  (skip {fn}: no report block yet)", file=sys.stderr)
            continue
        rows.append(dict(model=model, train_pct=pct, qf_n=qf_n, qh_n=qh_n,
                         qf_pre=qf_pre, qf_post=qf_post,
                         qh_pre=qh_pre, qh_post=qh_post,
                         qh_gain=round(qh_post - qh_pre, 4)))

    rows.sort(key=lambda r: (r["model"], r["train_pct"]))
    csv_path = os.path.join(OUT, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["model", "train_pct", "qf_n", "qh_n", "qf_pre",
                            "qf_post", "qh_pre", "qh_post", "qh_gain"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'model':<8} {'train':>6} {'Q_F':>5} {'Q_held':>7} "
          f"{'QF_pre->post':>14} {'Qh_pre->post':>14} {'Qh_gain':>8}")
    for r in rows:
        print(f"{r['model']:<8} {r['train_pct']:>5}% {r['qf_n']:>5} {r['qh_n']:>7} "
              f"{r['qf_pre']:>6.3f}->{r['qf_post']:<6.3f} "
              f"{r['qh_pre']:>6.3f}->{r['qh_post']:<6.3f} {r['qh_gain']:>+8.4f}")
    print(f"\nsummary.csv -> {csv_path}")


if __name__ == "__main__":
    main()
