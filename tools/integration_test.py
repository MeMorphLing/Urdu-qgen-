"""End-to-end smoke test on synthetic Urdu data (no HuggingFace download).

    python -m tools.integration_test

Exercises: make_pair/split_sentences -> TSV -> SentencePiece -> Dataset ->
training loop -> checkpoint -> greedy + beam -> BLEU/ROUGE/PPL -> samples.tsv.
Everything runs in a temp workspace; your real data/ and results/ are untouched.
"""
import csv
import os
import random
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prepare_data import make_pair, split_sentences

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NOUNS = ["\u062F\u0631\u06CC\u0627", "\u067E\u06C1\u0627\u0691", "\u0634\u06C1\u0631",
         "\u0642\u0644\u0639\u06C1", "\u0645\u0633\u062C\u062F", "\u0645\u06CC\u062F\u0627\u0646"]
PLACES = ["\u0644\u0627\u06C1\u0648\u0631", "\u06A9\u0631\u0627\u0686\u06CC",
          "\u0645\u0644\u062A\u0627\u0646", "\u067E\u0634\u0627\u0648\u0631"]
NUMS = ["3180", "1947", "250", "72", "1600"]


def synth_rows(n, seed=0):
    """Fabricate SQuAD-shaped rows so make_pair can be exercised for real."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        noun, place, num = rng.choice(NOUNS), rng.choice(PLACES), rng.choice(NUMS)
        if rng.random() < 0.5:
            ans, q = num, f"{noun} \u06A9\u062A\u0646\u0627 \u0644\u0645\u0628\u0627 \u06C1\u06D2\u061F"
            sent = f"{noun} \u0644\u06AF \u0628\u06BE\u06AF {ans} \u06A9\u0644\u0648\u0645\u06CC\u0679\u0631 \u0644\u0645\u0628\u0627 \u06C1\u06D2\u06D4"
        else:
            ans, q = place, f"{noun} \u06A9\u06C1\u0627\u06BA \u0648\u0627\u0642\u0639 \u06C1\u06D2\u061F"
            sent = f"{noun} {ans} \u0645\u06CC\u06BA \u0648\u0627\u0642\u0639 \u06C1\u06D2\u06D4"
        lead = "\u06CC\u06C1 \u0627\u06CC\u06A9 \u062A\u0639\u0627\u0631\u0641\u06CC \u062C\u0645\u0644\u06C1 \u06C1\u06D2\u06D4 "
        context = lead + sent + " \u0622\u062E\u0631\u06CC \u062C\u0645\u0644\u06C1\u06D4"
        rows.append({"context": context, "question": q,
                     "answers": {"text": [ans],
                                 "answer_start": [context.index(ans)]}})
    return rows


def write_tsv(path, rows):
    pairs = [p for p in (make_pair(r) for r in rows) if p is not None]
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE,
                   escapechar="\\").writerows(pairs)
    return pairs


def run(cmd, cwd):
    print(f"\n$ {' '.join(cmd)}")
    # Force UTF-8 in the child too: on Windows a piped stdout is cp1252 and the
    # child dies the moment it prints Urdu. errors="replace" means a decode
    # problem never masks the real error message.
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    print((r.stdout or "")[-2500:])
    if r.returncode != 0:
        print((r.stderr or "")[-4000:])
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout or ""


def main():
    # --- unit checks on the data prep primitives ---
    text = ("\u0627\u0644\u0641\u06D4 \u0628\u06D2\u061F \u062C\u06CC\u0645!")
    sents = list(split_sentences(text))
    assert len(sents) == 3, sents
    assert all(text[s:e] == snt for s, e, snt in sents)
    print("split_sentences: offsets are exact, 3 sentences found")

    rows = synth_rows(4, seed=1)
    src, tgt = make_pair(rows[0])
    assert "<ans>" in src and "</ans>" in src
    print("make_pair sample:", src, "||", tgt)

    bad = {"context": "\u0627\u0644\u0641\u06D4", "question": "\u06A9\u06CC\u0627\u061F",
           "answers": {"text": [], "answer_start": []}}
    assert make_pair(bad) is None
    print("unanswerable rows are dropped")

    ws = tempfile.mkdtemp(prefix="qgen_it_")
    for d in ["data", "tokenizer", "checkpoints", "results/figures", "configs"]:
        os.makedirs(os.path.join(ws, d), exist_ok=True)
    for name in ["src", "app", "tools"]:
        shutil.copytree(os.path.join(REPO, name), os.path.join(ws, name))

    train = write_tsv(os.path.join(ws, "data/train.tsv"), synth_rows(1500, 0))
    write_tsv(os.path.join(ws, "data/valid.tsv"), synth_rows(120, 7))
    write_tsv(os.path.join(ws, "data/wiki_test.tsv"), synth_rows(80, 9))
    print(f"\nwrote synthetic TSVs: {len(train)} train pairs")

    with open(os.path.join(REPO, "configs/base.yaml"), encoding="utf-8") as f:
        cfg = f.read()
    cfg = (cfg.replace("vocab_size: 8000", "vocab_size: 90")
              .replace("epochs: 15", "epochs: 3")
              .replace("batch_size: 64", "batch_size: 32")
              .replace("beam_subset: 2000", "beam_subset: 40")
              .replace("amp: true", "amp: false"))
    with open(os.path.join(ws, "configs/base.yaml"), "w", encoding="utf-8") as f:
        f.write(cfg)

    run([sys.executable, "-m", "src.train_tokenizer"], ws)
    run([sys.executable, "-m", "src.train", "--overfit", "32", "--epochs", "8"], ws)
    run([sys.executable, "-m", "src.train"], ws)
    out = run([sys.executable, "-m", "src.evaluate"], ws)
    run([sys.executable, "-m", "src.analyze"], ws)
    run([sys.executable, "-m", "src.viz", "--what", "loss"], ws)
    run([sys.executable, "-m", "src.human_eval", "--make"], ws)

    samples = os.path.join(ws, "results/samples.tsv")
    with open(samples, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) >= 50, f"samples.tsv has only {len(rows)} rows"
    assert set(rows[0]) == {"split", "source", "reference", "greedy", "beam"}
    print(f"\nsamples.tsv: {len(rows)} rows, columns OK")
    print("example:", rows[0]["source"], "->", rows[0]["beam"])

    assert "BLEU-4" in out
    assert "ROUGE-L" in out
    print(f"\nworkspace kept at {ws}")
    print("INTEGRATION TEST PASSED")


if __name__ == "__main__":
    main()
