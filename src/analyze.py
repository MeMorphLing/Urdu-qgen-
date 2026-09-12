"""Section 4.7 - discussion evidence.

    python -m src.analyze

Produces:
  * per-question-word accuracy (which interrogatives the model gets right)
  * sentence-BLEU delta between beam and greedy, so you can point at the cases
    where beam actually helps versus where it hurts
  * a copy-rate check (did the model just echo the source sentence?)
"""
import csv
import json
from collections import defaultdict

import sacrebleu

# Urdu interrogatives. First match in the question decides the bucket.
QWORDS = {
    "\u06A9\u06CC\u0627": "kya (what)",
    "\u06A9\u0648\u0646": "kaun (who)",
    "\u06A9\u0628": "kab (when)",
    "\u06A9\u06C1\u0627\u06BA": "kahan (where)",
    "\u06A9\u062A\u0646\u0627": "kitna (how much)",
    "\u06A9\u062A\u0646\u06CC": "kitni (how much)",
    "\u06A9\u062A\u0646\u06D2": "kitne (how many)",
    "\u06A9\u06CC\u0648\u06BA": "kyun (why)",
    "\u06A9\u06CC\u0633\u06D2": "kaise (how)",
    "\u06A9\u0648\u0646\u0633\u0627": "kaunsa (which)",
    "\u06A9\u0633": "kis (which/whom)",
}


def qword_of(question):
    toks = question.split()
    for t in toks:
        clean = t.strip("\u061F?\u06D4.,")
        if clean in QWORDS:
            return QWORDS[clean]
    return "other/none"


def sent_bleu(hyp, ref):
    return sacrebleu.sentence_bleu(hyp, [ref]).score


def main():
    with open("results/samples.tsv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    report = {}

    for split in sorted({r["split"] for r in rows}):
        sub = [r for r in rows if r["split"] == split]

        # --- question-word buckets ---
        buckets = defaultdict(lambda: {"n": 0, "match": 0, "bleu": 0.0})
        for r in sub:
            ref_q, hyp_q = qword_of(r["reference"]), qword_of(r["beam"])
            b = buckets[ref_q]
            b["n"] += 1
            b["match"] += int(ref_q == hyp_q)
            b["bleu"] += sent_bleu(r["beam"], r["reference"])
        qword_table = {
            k: {"n": v["n"],
                "qword_accuracy": round(100 * v["match"] / v["n"], 1),
                "mean_sentBLEU": round(v["bleu"] / v["n"], 2)}
            for k, v in sorted(buckets.items(), key=lambda x: -x[1]["n"])
        }

        # --- beam vs greedy ---
        deltas = [(sent_bleu(r["beam"], r["reference"])
                   - sent_bleu(r["greedy"], r["reference"]), r) for r in sub]
        better = sum(1 for d, _ in deltas if d > 1e-6)
        worse = sum(1 for d, _ in deltas if d < -1e-6)
        deltas.sort(key=lambda x: x[0])

        # --- degenerate behaviour ---
        def copied(r):
            src = r["source"].replace("<ans>", " ").replace("</ans>", " ").split()
            hyp = r["beam"].split()
            if not hyp:
                return False
            return len(set(hyp) & set(src)) / len(set(hyp)) > 0.9

        def repeats(r):
            t = r["beam"].split()
            return len(t) >= 4 and len(set(t)) / len(t) < 0.5

        report[split] = {
            "n": len(sub),
            "question_words": qword_table,
            "beam_vs_greedy": {
                "beam_better": better, "beam_worse": worse,
                "tied": len(sub) - better - worse,
                "mean_delta_sentBLEU": round(sum(d for d, _ in deltas) / len(sub), 3),
                "mean_len_greedy": round(sum(len(r["greedy"].split()) for r in sub) / len(sub), 2),
                "mean_len_beam": round(sum(len(r["beam"].split()) for r in sub) / len(sub), 2),
            },
            "degenerate": {
                "copied_source_pct": round(100 * sum(copied(r) for r in sub) / len(sub), 1),
                "repetition_pct": round(100 * sum(repeats(r) for r in sub) / len(sub), 1),
                "empty_pct": round(100 * sum(not r["beam"].strip() for r in sub) / len(sub), 1),
            },
        }

        print(f"\n=== {split} ===")
        print(json.dumps(report[split], indent=2, ensure_ascii=False))
        print("\n5 worst beam-vs-greedy cases (beam hurt most):")
        for d, r in deltas[:5]:
            print(f"  delta {d:+.1f} | ref: {r['reference']}")
            print(f"                | greedy: {r['greedy']}")
            print(f"                | beam  : {r['beam']}")

    with open("results/analysis.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nwrote results/analysis.json")


if __name__ == "__main__":
    main()
