"""Figures for Section 4.5.

    python -m src.viz --what loss
    python -m src.viz --what hist
    python -m src.viz --what attn --index 7

Urdu in matplotlib needs two things most people forget: a font that has the
glyphs, and arabic_reshaper + bidi so letters join and run right-to-left. If no
Urdu font is available we fall back to numeric tick labels plus a printed token
legend, which is ugly but readable - better than a grid of empty boxes.
"""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sentencepiece as spm
from matplotlib import font_manager

from src.common import get_device, load_cfg, load_checkpoint
from src.dataset import read_tsv
from src.decode import greedy_decode_one

FIG = "results/figures"
URDU_FONT_CANDIDATES = [
    "fonts/NotoNastaliqUrdu-Regular.ttf",
    "fonts/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Regular.ttf",
    "/usr/share/fonts/truetype/fonts-deva/NotoNaskhArabic-Regular.ttf",
]


def urdu_font():
    for p in URDU_FONT_CANDIDATES + glob.glob("fonts/*.ttf"):
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            return font_manager.FontProperties(fname=p)
    return None


def shape(text):
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except ImportError:
        return text


def plot_loss(log_csv):
    with open(log_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ep = [int(r["epoch"]) for r in rows]
    tl = [float(r["train_loss"]) for r in rows]
    vl = [float(r["valid_loss"]) for r in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(ep, tl, marker="o", label="train")
    plt.plot(ep, vl, marker="s", label="validation")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy per token")
    plt.title("Training and validation loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = f"{FIG}/loss_curve.png"
    plt.savefig(out, dpi=160)
    print("wrote", out)


def plot_hist(cfg):
    for name, path in [
        ("train", cfg["data"]["train_tsv"]),
        ("valid", cfg["data"]["valid_tsv"]),
    ]:
        pairs = read_tsv(path)
        src = [len(s.split()) for s, _ in pairs]
        tgt = [len(t.split()) for _, t in pairs]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].hist(src, bins=30)
        ax[0].set_title(f"{name}: source length (words)")
        ax[1].hist(tgt, bins=25, color="tab:orange")
        ax[1].set_title(f"{name}: target length (words)")
        for a in ax:
            a.set_xlabel("whitespace tokens")
            a.set_ylabel("count")
            a.grid(alpha=0.3)
        plt.tight_layout()
        out = f"{FIG}/length_hist_{name}.png"
        plt.savefig(out, dpi=160)
        print("wrote", out)
        plt.close(fig)


def plot_attention(cfg, index):
    device = get_device()
    model, _, _ = load_checkpoint(cfg["train"]["ckpt_path"], device)
    sp = spm.SentencePieceProcessor(model_file=cfg["tokenizer"]["model_file"])
    pairs = read_tsv(cfg["data"]["valid_tsv"])
    src_text, ref = pairs[index]

    src_ids = sp.encode(src_text)[: cfg["data"]["max_src_pieces"]]
    out_ids, attn = greedy_decode_one(model, src_ids, device, cfg["decode"]["max_len"])
    src_pieces = sp.id_to_piece(src_ids)
    out_pieces = sp.id_to_piece(out_ids)

    print("SOURCE   :", src_text)
    print("REFERENCE:", ref)
    print("GENERATED:", sp.decode(out_ids))

    fp = urdu_font()
    fig, ax = plt.subplots(
        figsize=(max(6, len(src_pieces) * 0.45), max(4, len(out_pieces) * 0.45))
    )
    im = ax.imshow(attn[: len(out_pieces)], aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(src_pieces)))
    ax.set_yticks(range(len(out_pieces)))
    if fp is not None:
        ax.set_xticklabels(
            [shape(p.replace("\u2581", " ")) for p in src_pieces],
            rotation=90,
            fontproperties=fp,
        )
        ax.set_yticklabels(
            [shape(p.replace("\u2581", " ")) for p in out_pieces], fontproperties=fp
        )
    else:
        ax.set_xticklabels(range(len(src_pieces)), rotation=90)
        ax.set_yticklabels(range(len(out_pieces)))
        print("\nno Urdu font found - axis legend:")
        print("source :", list(enumerate(src_pieces)))
        print("output :", list(enumerate(out_pieces)))
    ax.set_xlabel("source pieces")
    ax.set_ylabel("generated pieces")
    fig.colorbar(im, ax=ax, shrink=0.7)
    plt.title("Attention weights")
    plt.tight_layout()
    out = f"{FIG}/attention_{index}.png"
    plt.savefig(out, dpi=160)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--what", choices=["loss", "hist", "attn", "all"], default="all")
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    os.makedirs(FIG, exist_ok=True)

    if args.what in ("loss", "all"):
        plot_loss(cfg["train"]["log_csv"])
    if args.what in ("hist", "all"):
        plot_hist(cfg)
    if args.what in ("attn", "all"):
        plot_attention(cfg, args.index)


if __name__ == "__main__":
    main()
