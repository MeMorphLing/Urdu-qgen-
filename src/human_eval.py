"""Section 3.2 - human evaluation.

Step 1, make the rating sheets (fixed seed, 50 validation outputs):
    python -m src.human_eval --make

    -> results/human_eval_member1.csv
    -> results/human_eval_member2.csv

Each member fills fluency / relevance / answerability with 1 or 0, ALONE.
Rate independently and only compare afterwards - rating together collapses the
disagreement that Cohen's kappa is meant to measure.

Step 2, score them:
    python -m src.human_eval --score
"""
import argparse
import csv
import random

from sklearn.metrics import cohen_kappa_score

from src.common import load_cfg

CRITERIA = ["fluency", "relevance", "answerability"]
SHEETS = ["results/human_eval_member1.csv", "results/human_eval_member2.csv"]


def make_sheets(cfg, n=50, split="uqa_valid"):
    with open("results/samples.tsv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rows = [r for r in rows if r["split"] == split]
    rng = random.Random(cfg["seed"])
    sample = rng.sample(rows, min(n, len(rows)))

    for path in SHEETS:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "source", "reference", "generated_beam"] + CRITERIA)
            for i, r in enumerate(sample):
                w.writerow([i, r["source"], r["reference"], r["beam"], "", "", ""])
        print("wrote", path)
    print(f"\n{len(sample)} items, seed {cfg['seed']}. Both members rate 1/0 "
          f"independently, then run --score.")


def load_sheet(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for c in CRITERIA:
        vals = []
        for r in rows:
            v = (r[c] or "").strip()
            if v not in ("0", "1"):
                raise SystemExit(f"{path}: row {r['id']} has empty/invalid '{c}'")
            vals.append(int(v))
        out[c] = vals
    return out


def score_sheets():
    m1, m2 = load_sheet(SHEETS[0]), load_sheet(SHEETS[1])
    print("\n=== Table 4 - human evaluation (50 samples) ===")
    header = f"{'':<18}" + "".join(f"{c:>16}" for c in CRITERIA)
    print(header)
    for name, m in [("Member 1 (% yes)", m1), ("Member 2 (% yes)", m2)]:
        line = f"{name:<18}"
        for c in CRITERIA:
            line += f"{100*sum(m[c])/len(m[c]):>15.1f}%"
        print(line)
    line = f"{'Cohen kappa':<18}"
    rows = []
    for c in CRITERIA:
        k = cohen_kappa_score(m1[c], m2[c])
        line += f"{k:>16.3f}"
        rows.append([c, round(100*sum(m1[c])/len(m1[c]), 1),
                     round(100*sum(m2[c])/len(m2[c]), 1), round(float(k), 3)])
    print(line)

    with open("results/human_eval_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["criterion", "member1_pct_yes", "member2_pct_yes", "cohen_kappa"])
        w.writerows(rows)
    print("\nwrote results/human_eval_summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    if args.make:
        make_sheets(cfg, args.n)
    if args.score:
        score_sheets()
    if not (args.make or args.score):
        ap.print_help()


if __name__ == "__main__":
    main()
