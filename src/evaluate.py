"""Task 4 - evaluation on UQA validation and Wiki-UQA.

    python -m src.evaluate --config configs/base.yaml

CRITICAL: rouge_score's default tokenizer strips every non-[a-z0-9] character,
which deletes all Urdu and silently returns ROUGE-L = 0.0. We pass a whitespace
tokenizer instead. sacrebleu is reported with both the default '13a' tokenizer
(comparable with published SQuAD-QG numbers) and 'intl'.
"""

import argparse
import csv
import json
import time

import sacrebleu
import sentencepiece as spm
from rouge_score import rouge_scorer
from torch import nn

from src.common import (
    PAD,
    evaluate_loss,
    get_device,
    load_cfg,
    load_checkpoint,
    set_seed,
)
from src.dataset import QGDataset, make_loader
from src.decode import beam_search_one, greedy_decode_batch

UNK_CHAR = "\u2047"  # SentencePiece decodes <unk> as U+2047


class WhitespaceTokenizer:
    """Keeps Urdu characters intact; rouge_score's default would delete them."""

    def tokenize(self, text):
        return text.split()


def rouge_l(hyps, refs):
    scorer = rouge_scorer.RougeScorer(
        ["rougeL"], use_stemmer=False, tokenizer=WhitespaceTokenizer()
    )
    total = sum(scorer.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs))
    return total / max(1, len(refs))


def unk_rate(hyps):
    unk = sum(h.count(UNK_CHAR) for h in hyps)
    tok = sum(len(h.split()) for h in hyps)
    return unk / max(1, tok)


def score(hyps, refs):
    return {
        "BLEU-4": round(sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a").score, 2),
        "BLEU-4_intl": round(
            sacrebleu.corpus_bleu(hyps, [refs], tokenize="intl").score, 2
        ),
        "ROUGE-L": round(rouge_l(hyps, refs), 4),
        "unk_rate": round(unk_rate(hyps), 5),
        "mean_hyp_len": round(sum(len(h.split()) for h in hyps) / max(1, len(hyps)), 2),
        "n": len(hyps),
    }


def run_split(model, sp, cfg, tsv_path, device, split_name, results, samples_rows):
    d, dec = (
        cfg["data"],
        cfg["decode"],
    )  # Local variable `pairs` is assigned to but never used
    ds = QGDataset(tsv_path, sp, d["max_src_pieces"], d["max_tgt_pieces"])
    loader = make_loader(ds, cfg["train"]["batch_size"], shuffle=False)

    # ---- perplexity (teacher forced, token level) ----
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    _, ppl = evaluate_loss(model, loader, criterion, device)

    # ---- greedy over the whole split ----
    t0 = time.time()
    greedy_hyps, refs, srcs = [], [], []
    for src, src_len, _, _ in loader:
        outs = greedy_decode_batch(model, src, src_len, device, dec["max_len"])
        greedy_hyps.extend(sp.decode(o) for o in outs)
    for i in range(len(ds)):
        srcs.append(ds.pairs[i][0])
        refs.append(ds.pairs[i][1])
    refs = refs[: len(greedy_hyps)]
    srcs = srcs[: len(greedy_hyps)]
    print(f"{split_name}: greedy done in {time.time() - t0:.0f}s")

    g = score(greedy_hyps, refs)
    g["PPL"] = round(ppl, 2)
    results[f"{split_name}/greedy"] = g

    # ---- beam over a fixed subset (beam is batch-1 and slow) ----
    n_beam = dec["beam_subset"] or len(ds)
    n_beam = min(n_beam, len(ds))
    t0 = time.time()
    beam_hyps = []
    for i in range(n_beam):
        ids = ds.data[i][0]
        out = beam_search_one(
            model, ids, device, dec["beam_size"], dec["max_len"], dec["length_alpha"]
        )
        beam_hyps.append(sp.decode(out))
    print(
        f"{split_name}: beam({dec['beam_size']}) x{n_beam} in {time.time() - t0:.0f}s"
    )

    b = score(beam_hyps, refs[:n_beam])
    b["PPL"] = round(ppl, 2)
    b["beam_size"] = dec["beam_size"]
    b["length_alpha"] = dec["length_alpha"]
    results[f"{split_name}/beam"] = b

    for i in range(min(n_beam, len(greedy_hyps))):
        samples_rows.append(
            [split_name, srcs[i], refs[i], greedy_hyps[i], beam_hyps[i]]
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    cfg_file = load_cfg(args.config)
    set_seed(cfg_file["seed"])
    device = get_device()
    ckpt_path = args.ckpt or cfg_file["train"]["ckpt_path"]

    model, cfg, ckpt = load_checkpoint(ckpt_path, device)
    cfg["decode"] = cfg_file["decode"]  # let config changes apply at eval time
    cfg["data"] = cfg_file["data"]
    cfg["train"] = cfg_file["train"]
    print(
        f"loaded {ckpt_path} (epoch {ckpt['epoch']}, val loss {ckpt['val_loss']:.4f})"
    )

    sp = spm.SentencePieceProcessor(model_file=cfg_file["tokenizer"]["model_file"])

    results, samples_rows = {}, []
    run_split(
        model,
        sp,
        cfg,
        cfg_file["data"]["valid_tsv"],
        device,
        "uqa_valid",
        results,
        samples_rows,
    )
    run_split(
        model,
        sp,
        cfg,
        cfg_file["data"]["wiki_tsv"],
        device,
        "wiki_uqa",
        results,
        samples_rows,
    )

    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open("results/samples.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["split", "source", "reference", "greedy", "beam"])
        w.writerows(samples_rows)

    print("\n=== Table 3 ===")
    print(f"{'split':<12}{'decode':<8}{'BLEU-4':>8}{'ROUGE-L':>9}{'PPL':>8}{'unk%':>8}")
    for key, v in results.items():
        sp_name, dec_name = key.split("/")
        print(
            f"{sp_name:<12}{dec_name:<8}{v['BLEU-4']:>8}{v['ROUGE-L']:>9}"
            f"{v['PPL']:>8}{v['unk_rate'] * 100:>8.3f}"
        )
    print("\nwrote results/metrics.json and results/samples.tsv")


if __name__ == "__main__":
    main()
