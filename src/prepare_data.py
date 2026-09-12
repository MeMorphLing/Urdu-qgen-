"""Task 1 - Data preparation.

Loads uqa/UQA (and uqa/Wiki-UQA for out-of-domain testing), keeps answerable
rows, locates the sentence containing the answer via character offsets, wraps
the answer in <ans>...</ans>, applies the length filter and writes TSVs.

Usage:
    python -m src.prepare_data --config configs/base.yaml
    python -m src.prepare_data --config configs/base.yaml --stream

With --stream the dataset is never cached to disk; rows arrive over HTTP and
only the output TSVs (~25 MB total) are written.
"""
import argparse
import csv
import json
import os
from collections import Counter

from src.common import ensure_dirs, load_cfg

ANS_OPEN, ANS_CLOSE = "<ans>", "</ans>"

# Urdu full stop U+06D4, Urdu question mark U+061F, exclamation mark.
# UQA is machine translated from SQuAD, so a sizeable minority of contexts keep
# ASCII punctuation. We add '.', '?' and newline as fallback delimiters; any bad
# split is caught by the offset-consistency check below and dropped.
SENT_DELIMS = "\u06D4\u061F!.?\n"


def split_sentences(text):
    """Yield (start, end, sentence) with character offsets into `text`."""
    start = 0
    for i, ch in enumerate(text):
        if ch in SENT_DELIMS:
            yield start, i + 1, text[start:i + 1]
            start = i + 1
    if start < len(text):
        yield start, len(text), text[start:]


def normalize_answer(example):
    """Return (answer_text, answer_start) or None if the row is unanswerable.

    The UQA hub dataset uses FLAT columns - answer (str), answer_start (int),
    is_impossible (bool) - not the nested SQuAD-style
    answers={"text": [...], "answer_start": [...]} that the assignment
    appendix assumes. We accept either, plus the list-of-dicts variant, so the
    same code works whichever shape the hub serves.
    """
    if example.get("is_impossible"):
        return None

    # flat schema (what uqa/UQA actually ships)
    if "answer" in example and example.get("answer_start") is not None:
        return example["answer"], example["answer_start"]

    answers = example.get("answers")

    # nested SQuAD schema: {"text": [...], "answer_start": [...]}
    if isinstance(answers, dict):
        texts = answers.get("text") or []
        starts = answers.get("answer_start") or []
        if texts and starts:
            return texts[0], starts[0]
        return None

    # list-of-dicts schema: [{"text": ..., "answer_start": ...}, ...]
    if isinstance(answers, list) and answers:
        a = answers[0]
        if isinstance(a, dict) and a.get("text") is not None:
            return a["text"], a.get("answer_start")
        return None

    return None


def make_pair(example, max_src=60, max_tgt=25, stats=None):
    """Return (source, target) or None if the row is unusable."""
    def bump(key):
        if stats is not None:
            stats[key] += 1

    found = normalize_answer(example)
    if found is None:
        bump("drop_unanswerable")
        return None

    a_text, a_start = found
    if a_start is None or a_start < 0 or not a_text or not a_text.strip():
        bump("drop_bad_offset")
        return None

    context = example["context"]
    question = " ".join((example["question"] or "").split())
    if not question:
        bump("drop_empty_question")
        return None

    for s, e, sent in split_sentences(context):
        if s <= a_start < e:
            rel = a_start - s
            if sent[rel:rel + len(a_text)] != a_text:
                bump("drop_offset_mismatch")
                return None
            src = (sent[:rel] + " " + ANS_OPEN + " " + a_text + " " +
                   ANS_CLOSE + " " + sent[rel + len(a_text):]).strip()
            src = " ".join(src.split())
            if len(src.split()) > max_src:
                bump("drop_src_too_long")
                return None
            if len(question.split()) > max_tgt:
                bump("drop_tgt_too_long")
                return None
            bump("kept")
            return src, question
    bump("drop_no_sentence_found")
    return None


def build_split(split, out_path, max_src, max_tgt):
    stats = Counter()
    seen, pairs = set(), []
    n_raw = 0
    for ex in split:
        n_raw += 1
        if n_raw % 20000 == 0:
            print(f"  ...{n_raw} rows read, {len(pairs)} pairs kept")
        p = make_pair(ex, max_src, max_tgt, stats)
        if p is None:
            continue
        if p in seen:                     # exact duplicate pairs add nothing
            stats["drop_duplicate"] += 1
            continue
        seen.add(p)
        pairs.append(p)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerows(pairs)

    src_lens = [len(s.split()) for s, _ in pairs]
    tgt_lens = [len(t.split()) for _, t in pairs]
    summary = {
        "out_path": out_path,
        "raw_rows": n_raw,
        "answerable_rows": n_raw - stats["drop_unanswerable"],
        "pairs_after_filter": len(pairs),
        "mean_src_len": round(sum(src_lens) / max(1, len(src_lens)), 2),
        "mean_tgt_len": round(sum(tgt_lens) / max(1, len(tgt_lens)), 2),
        "drop_reasons": dict(stats),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return pairs, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--stream", action="store_true",
                    help="stream from HuggingFace instead of caching the full "
                         "dataset to disk; only the output TSVs are written")
    args = ap.parse_args()

    from datasets import load_dataset   # imported here so the helpers above
                                       # stay usable without the HF library
    cfg = load_cfg(args.config)
    ensure_dirs()
    d = cfg["data"]
    ms, mt = d["max_src_words"], d["max_tgt_words"]

    mode = "streaming" if args.stream else "cached"
    print(f"loading {d['hf_dataset']} ({mode}) ...")
    ds = load_dataset(d["hf_dataset"], streaming=args.stream)
    if not args.stream:
        print(ds)

    stats = {}
    _, stats["train"] = build_split(ds["train"], d["train_tsv"], ms, mt)
    valid_key = "validation" if "validation" in ds else "test"
    _, stats["valid"] = build_split(ds[valid_key], d["valid_tsv"], ms, mt)

    print(f"loading {d['hf_ood_dataset']} ({mode}) ...")
    wiki = load_dataset(d["hf_ood_dataset"], streaming=args.stream)
    wiki_key = next(k for k in ["test", "validation", "train"] if k in wiki)
    _, stats["wiki"] = build_split(wiki[wiki_key], d["wiki_tsv"], ms, mt)

    with open("results/dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("\nwrote results/dataset_stats.json  ->  fills Table 1")


if __name__ == "__main__":
    main()