"""Task 2 - Tokenizer.

Trains a shared 8k unigram SentencePiece model on TRAIN sources + targets only
(never valid, never Wiki-UQA - that would be leakage), registers <ans>/</ans>
as user-defined symbols, and verifies the model is sane before anything
downstream depends on it.

Usage:
    python -m src.train_tokenizer --config configs/base.yaml
"""
import argparse
import csv
import random

import sentencepiece as spm

from src.common import ANS_CLOSE, ANS_OPEN, ensure_dirs, load_cfg


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        return [(row[0], row[1]) for row in r if len(row) >= 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    ensure_dirs()
    d, t = cfg["data"], cfg["tokenizer"]

    train_pairs = read_tsv(d["train_tsv"])
    print(f"{len(train_pairs)} training pairs")

    with open(d["sp_corpus"], "w", encoding="utf-8") as f:
        f.writelines(f"{src}\n{tgt}\n" for src, tgt in train_pairs)

    spm.SentencePieceTrainer.train(
        input=d["sp_corpus"],
        model_prefix=t["model_prefix"],
        vocab_size=t["vocab_size"],
        model_type=t["model_type"],
        character_coverage=t["character_coverage"],
        user_defined_symbols=[ANS_OPEN, ANS_CLOSE],
        pad_id=t["pad_id"], unk_id=t["unk_id"],
        bos_id=t["bos_id"], eos_id=t["eos_id"],
    )

    sp = spm.SentencePieceProcessor(model_file=t["model_file"])

    # --- assertions: catch a broken tokenizer now, not after a 2-hour train ---
    # Note: encoding a BARE tag returns ['_', '<ans>'] because SentencePiece's
    # add_dummy_prefix prepends a word-start marker. That is normal. What
    # matters is that the tag survives as ONE piece inside a real sentence.
    for tag in (ANS_OPEN, ANS_CLOSE):
        assert sp.piece_to_id(tag) >= 4, f"{tag} is not in the vocabulary"
        assert sp.encode(tag, out_type=str)[-1] == tag, \
            f"{tag} was not registered as an atomic symbol"
    probe = next(s for s, _ in train_pairs if ANS_OPEN in s)
    pieces = sp.encode(probe, out_type=str)
    assert pieces.count(ANS_OPEN) == 1 and pieces.count(ANS_CLOSE) == 1, \
        f"tags were split inside a real sentence: {pieces}"
    assert sp.pad_id() == 0 and sp.unk_id() == 1 and sp.bos_id() == 2 and sp.eos_id() == 3

    # Round-trip: nmt_nfkc normalisation is lossy by design, so exact equality
    # is the wrong invariant. NFKC decomposes U+2047 (which UQA itself contains,
    # left over from its translation pipeline) into "??", among other mappings.
    # What must hold is IDEMPOTENCE: encoding stabilises after one pass.
    rng = random.Random(0)
    sample = rng.sample(train_pairs, min(200, len(train_pairs)))
    mismatches = []
    for src, tgt in sample:
        for text in (src, tgt):
            once = sp.decode(sp.encode(text))
            twice = sp.decode(sp.encode(once))
            assert once == twice, f"tokenizer is not idempotent on: {text}"
            if once != text:
                mismatches.append((text, once))

    rate = len(mismatches) / (2 * len(sample))
    print(f"round-trip: {100 * (1 - rate):.1f}% byte-exact; "
          f"{len(mismatches)} differ by NFKC normalisation only")
    for orig, got in mismatches[:2]:
        print("  normalised:", orig[:70])
        print("          ->:", got[:70])
    assert rate < 0.05, f"{100*rate:.1f}% of samples changed under normalisation " \
                        f"- that is too many, inspect the corpus"
    print("tokenizer assertions passed (atomic tags, reserved ids, idempotent)")

    # --- Task 2.3: five tokenised examples for the report ---
    print("\n=== five tokenised examples (Task 2.3) ===")
    for src, tgt in rng.sample(train_pairs, 5):
        print("\nSRC:", src)
        print("PIECES:", sp.encode(src, out_type=str))
        print("TGT:", tgt)
        print("PIECES:", sp.encode(tgt, out_type=str))
        print("IDS:", sp.encode(tgt))

    lens = [len(sp.encode(s)) for s, _ in train_pairs[:5000]]
    print(f"\nmean source pieces (first 5k): {sum(lens)/len(lens):.1f}")
    print(f"vocab size: {sp.get_piece_size()}")


if __name__ == "__main__":
    main()