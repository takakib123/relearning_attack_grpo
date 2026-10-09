"""
plot_figure4.py
===============
Paper Figure 4: TOFU original-question protocol, held-out direct-match rate after
SFT and GRPO relearning, as a function of the relearning portion.

Reads the summary.csv files written by collect_sweep.py for both sweeps plus a
baselines CSV (columns: model, unlearned_base, full_ceiling). By default it reads
the bundled paper numbers under paper_results/figure4/.

  python jobs/plot_figure4.py                                   # paper numbers
  python jobs/plot_figure4.py --grpo results/grpo_frac_sweep/summary.csv \
                              --sft results/sft_frac_sweep/summary.csv
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(os.path.dirname(HERE), "paper_results", "figure4")

COLORS = {"rmu": "#b07c00", "simnpo": "#1f4fa8", "altpo": "#1b7a4a"}
NAME = {"rmu": "RMU", "simnpo": "SimNPO", "altpo": "AltPO"}
ORDER = ["rmu", "simnpo", "altpo"]


def load_sweep(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(r["model"], {})[int(r["train_pct"])] = float(r["qh_post"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grpo", default=os.path.join(PAPER, "grpo_frac_sweep_summary.csv"))
    ap.add_argument("--sft", default=os.path.join(PAPER, "sft_frac_sweep_summary.csv"))
    ap.add_argument("--baselines", default=os.path.join(PAPER, "baselines.csv"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "figure4.pdf"))
    args = ap.parse_args()

    grpo, sft = load_sweep(args.grpo), load_sweep(args.sft)
    base = {r["model"]: r for r in csv.DictReader(open(args.baselines))}
    models = [m for m in ORDER if m in grpo and m in sft] + sorted(
        m for m in set(grpo) & set(sft) if m not in ORDER)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = [20, 40, 60, 80]
    ceiling = float(next(iter(base.values()))["full_ceiling"])
    ax.axhline(ceiling, color="gray", ls="--", lw=1, label="pre-unlearning baseline")
    for m in models:
        c = COLORS.get(m, None)
        ax.plot(xs, [sft[m][x] for x in xs], "-o", color=c, label=f"{NAME.get(m, m)} — SFT")
        ax.plot(xs, [grpo[m][x] for x in xs], "--^", color=c, alpha=0.6,
                label=f"{NAME.get(m, m)} — GRPO")
        if m in base:
            ax.axhline(float(base[m]["unlearned_base"]), color=c, ls=":", lw=1)
        ax.annotate(f"{NAME.get(m, m)}\nSFT {sft[m][80]:.2f} GRPO {grpo[m][80]:.2f}",
                    (80, sft[m][80]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=c, fontsize=8)
    ax.set_xticks(xs)
    ax.set_xlim(14, 100)
    ax.set_ylim(0, max(ceiling * 1.05, 0.85))
    ax.set_xlabel("Relearning attack portion (%)")
    ax.set_ylabel("Match rate")
    ax.set_title("SFT vs. GRPO relearning — held-out recall", loc="left", fontweight="bold")
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper center")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
