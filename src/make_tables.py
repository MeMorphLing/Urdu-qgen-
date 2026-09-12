"""Report tables 1-4 as markdown, ready to paste into README.md and the blog.

    python -m src.make_tables
    python -m src.make_tables --results-dir results --ckpt checkpoints/best.pt

Presentation only: every number here is read from an artifact the pipeline
already wrote. Nothing is recomputed, so these tables can never disagree with
results/metrics.json. Missing artifacts are reported in place and the remaining
tables still print, which means this is safe to run halfway through the run.
"""

import argparse
import csv
import json
import os
import pickle

import torch

from src.common import configure_stdout

configure_stdout()

# What to tell the user to run when an artifact isn't there yet.
PRODUCED_BY = {
    "dataset_stats.json": "python -m src.prepare_data --config configs/base.yaml",
    "metrics.json": "python -m src.evaluate --config configs/base.yaml",
    "human_eval_summary.csv": "python -m src.human_eval --score",
    "train_log.csv": "python -m src.train --config configs/base.yaml",
    "best.pt": "python -m src.train --config configs/base.yaml",
}

SPLIT_NAMES = {"uqa_valid": "UQA valid", "wiki_uqa": "Wiki-UQA"}


def missing(path):
    """Markdown note standing in for an artifact that doesn't exist yet."""
    cmd = PRODUCED_BY.get(os.path.basename(path), "the pipeline step that writes it")
    return f"*`{path.replace(os.sep, '/')}` not yet generated — run `{cmd}`*"


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_ckpt(path):
    if not os.path.exists(path):
        return None
    # map_location='cpu' so this works on a laptop with no GPU; the state_dict
    # is only ever measured, never loaded into a module.
    return torch.load(path, map_location="cpu", weights_only=False)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def num(value, fmt="{}"):
    return "—" if value is None else fmt.format(value)


def table1(stats):
    if stats is None:
        return None
    cols = [("Train", "train"), ("Validation", "valid"), ("Wiki-UQA", "wiki")]

    def cell(key, field, fmt="{:,}"):
        v = (stats.get(key) or {}).get(field)
        return "—" if v is None else fmt.format(v)

    def lengths(key):
        s = (stats.get(key) or {})
        if s.get("mean_src_len") is None or s.get("mean_tgt_len") is None:
            return "—"
        return f"{s['mean_src_len']} / {s['mean_tgt_len']}"

    rows = [
        ["Rows in raw dataset"] + [cell(k, "raw_rows") for _, k in cols],
        ["Answerable rows"] + [cell(k, "answerable_rows") for _, k in cols],
        ["Pairs after length filter"] + [cell(k, "pairs_after_filter") for _, k in cols],
        ["Mean source / target length (words)"] + [lengths(k) for _, k in cols],
    ]
    return table([""] + [name for name, _ in cols], rows)


def table2(ckpt, ckpt_path, log_rows, log_path):
    rows = []
    notes = []

    if ckpt is None:
        notes.append(missing(ckpt_path))
    else:
        cfg = ckpt.get("cfg") or {}
        m, tr = cfg.get("model") or {}, cfg.get("train") or {}
        rnn = str(m.get("rnn_type", "rnn")).upper()
        n_params = sum(t.numel() for t in (ckpt.get("model") or {}).values()
                       if torch.is_tensor(t))
        rows += [
            ["Encoder / decoder type",
             (
                 f"{num(m.get('enc_layers'))}-layer Bi{rnn} / "
                 f"{num(m.get('dec_layers'))}-layer {rnn}"
             )],
            ["Attention", "Bahdanau (additive), masked at PAD"],
            ["Embedding dim", num(m.get("emb_dim"))],
            ["Hidden dim", num(m.get("hid_dim"))],
            ["Attention dim", num(m.get("attn_dim"))],
            ["Dropout", num(m.get("dropout"))],
            ["Vocabulary", num(ckpt.get("vocab_size"), "{:,}") + " (shared, unigram SentencePiece)"],
            ["Trainable parameters", f"{n_params:,}"],
            ["Optimiser",
             (
                 f"Adam, lr {num(tr.get('lr'))}, ReduceLROnPlateau "
                 f"(x{num(tr.get('lr_factor'))}, patience {num(tr.get('lr_patience'))})"
             )],
            ["Batch size", num(tr.get("batch_size"))],
        ]

    if log_rows is None:
        notes.append(missing(log_path))
    else:
        secs = sum(float(r["secs"]) for r in log_rows if r.get("secs"))
        rows += [
            ["Epochs actually run", len(log_rows)],
            ["Total wall-clock", fmt_secs(secs)],
        ]

    if not rows:
        return "\n\n".join(notes)
    md = table(["", ""], rows)
    return md + ("\n\n" + "\n\n".join(notes) if notes else "")


def fmt_secs(secs):
    if secs >= 3600:
        return f"{int(secs // 3600)} h {int(secs % 3600 // 60)} min"
    return f"{secs / 60:.1f} min"


def table3(metrics):
    if metrics is None:
        return None
    rows = []
    for key, v in metrics.items():
        split, decoding = key.split("/", 1) if "/" in key else (key, "")
        if decoding == "beam" and v.get("beam_size"):
            decoding = f"beam (k={v['beam_size']})"
        unk = v.get("unk_rate")
        rows.append([
            SPLIT_NAMES.get(split, split),
            decoding,
            num(v.get("BLEU-4")),
            num(v.get("ROUGE-L")),
            num(v.get("PPL")),
            num(None if unk is None else unk * 100, "{:.3f}"),
        ])
    return table(["Split", "Decoding", "BLEU-4", "ROUGE-L", "PPL", "`<unk>` %"], rows)


def table4(human):
    if human is None:
        return None
    criteria = [r["criterion"] for r in human]
    rows = [
        ["Member 1 (% yes)"] + [num(float(r["member1_pct_yes"]), "{:.1f}%") for r in human],
        ["Member 2 (% yes)"] + [num(float(r["member2_pct_yes"]), "{:.1f}%") for r in human],
        ["Cohen's kappa"] + [num(float(r["cohen_kappa"]), "{:.3f}") for r in human],
    ]
    return table([""] + [c.capitalize() for c in criteria], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    args = ap.parse_args()

    rd = args.results_dir
    stats_path = os.path.join(rd, "dataset_stats.json")
    metrics_path = os.path.join(rd, "metrics.json")
    human_path = os.path.join(rd, "human_eval_summary.csv")
    log_path = os.path.join(rd, "train_log.csv")

    try:
        ckpt = load_ckpt(args.ckpt)
    except (FileNotFoundError, OSError, EOFError, RuntimeError, pickle.UnpicklingError) as e:
        print(f"could not read {args.ckpt}: {e}")
        ckpt = None

    blocks = [
        ("Table 1 — Dataset statistics", table1(load_json(stats_path)), stats_path),
        ("Table 2 — Model configuration",
         table2(ckpt, args.ckpt, load_csv(log_path), log_path), None),
        ("Table 3 — Automatic metrics", table3(load_json(metrics_path)), metrics_path),
        ("Table 4 — Human evaluation", table4(load_csv(human_path)), human_path),
    ]

    out = []
    for title, md, path in blocks:
        out.append(f"## {title}\n")
        out.append((md if md is not None else missing(path)) + "\n")
    text = "\n".join(out)

    print(text)
    os.makedirs(rd, exist_ok=True)
    dest = os.path.join(rd, "tables.md")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
