"""Collect relearn.py outputs into the WMDP table.

    python scripts/summarize.py results/table4          # the paper's runs
    python scripts/summarize.py results/runs --plateau  # report plateau instead

Attack accuracy is held-out (D_e) letter accuracy; by default the best epoch,
with --plateau the mean of the last four evaluations. MMLU utility is the
final-epoch retain accuracy. "Unlearned" is the epoch-0 evaluation of the
attacked checkpoint. "Pre" is filled from evaluation-only runs (--epochs 0) of
the base models when present in the same directory.
"""
import argparse
import glob
import json
import os

import numpy as np

# unlearned checkpoint -> (display name, base model)
CHECKPOINTS = {
    "OPTML-Group/SimNPO-WMDP-zephyr-7b-beta": ("SimNPO (Zephyr-7B-beta)", "HuggingFaceH4/zephyr-7b-beta"),
    "OPTML-Group/NPO-SAM-WMDP": ("NPO-SAM (Zephyr-7B-beta)", "HuggingFaceH4/zephyr-7b-beta"),
    "lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2": ("TAR-Bio-v2 (Llama-3-8B-Instruct)", "NousResearch/Meta-Llama-3-8B-Instruct"),
    "OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct": ("IDK-AP (Llama-3-8B-Instruct)", "NousResearch/Meta-Llama-3-8B-Instruct"),
}


def load_runs(dirs):
    runs = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                r = json.load(open(path))
                hist, args = r["history"], r["args"]
            except (KeyError, json.JSONDecodeError):
                continue
            mode = "eval" if args.get("epochs") == 0 else args.get("mode") or "grpo"
            runs.append({"path": path, "model": r["model"], "mode": mode, "hist": hist})
    return runs


def attack_score(hist, plateau):
    ho = [e["heldout_raw"] for e in hist]
    return float(np.mean(ho[-4:])) if plateau else max(ho)


def fmt(x):
    return "  -  " if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--plateau", action="store_true")
    args = ap.parse_args()

    runs = load_runs(args.dirs)
    if not runs:
        print("no runs found")
        return

    pre = {r["model"]: r["hist"][0] for r in runs if r["mode"] == "eval"}
    by_key = {(r["model"], r["mode"]): r for r in runs}

    hdr = (f'{"Unlearning method (base LLM)":<36}'
           f'{"Pre":>7}{"Unl.":>7}{"SFT":>7}{"GRPO":>7}   |'
           f'{"Pre":>7}{"Unl.":>7}{"SFT":>7}{"GRPO":>7}')
    print(f'{"":<36}{"Attack accuracy":^28}   |{"MMLU utility":^28}')
    print(hdr)
    print("-" * len(hdr))
    for model, (name, base) in CHECKPOINTS.items():
        grpo, sft = by_key.get((model, "grpo")), by_key.get((model, "sft-full"))
        start = (grpo or sft)["hist"][0] if (grpo or sft) else None
        p = pre.get(base)
        acc = [p and p["heldout_raw"], start and start["heldout_raw"],
               sft and attack_score(sft["hist"], args.plateau),
               grpo and attack_score(grpo["hist"], args.plateau)]
        util = [p and p["retain_raw"], start and start["retain_raw"],
                sft and sft["hist"][-1]["retain_raw"],
                grpo and grpo["hist"][-1]["retain_raw"]]
        print(f"{name:<36}" + "".join(f"{fmt(x):>7}" for x in acc) + "   |"
              + "".join(f"{fmt(x):>7}" for x in util))

    print("\nPer-run trajectories (held-out accuracy by epoch):")
    for r in runs:
        ho = " ".join(f"{e['heldout_raw']:.3f}" for e in r["hist"])
        print(f"  {os.path.basename(r['path']):<60} {r['mode']:<9} {ho}")


if __name__ == "__main__":
    main()
