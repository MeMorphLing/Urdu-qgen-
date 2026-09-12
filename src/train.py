"""Task 3 - training.

    python -m src.train --config configs/base.yaml
    python -m src.train --limit 10000 --epochs 1        # smoke run
    python -m src.train --overfit 32                    # sanity: loss -> ~0

Logs per-epoch train/valid loss and perplexity to results/train_log.csv and
keeps the best checkpoint by validation loss.
"""
import argparse
import csv
import math
import time

import sentencepiece as spm
import torch
from torch import nn

from src.common import (
    PAD,
    build_model,
    count_params,
    ensure_dirs,
    evaluate_loss,
    get_device,
    load_cfg,
    save_checkpoint,
    set_seed,
)
from src.dataset import QGDataset, make_loader


def train_one_epoch(model, loader, optimizer, criterion, device, clip, scaler=None):
    model.train()
    total_loss, total_tokens = 0.0, 0
    for src, src_len, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, _ = model(src, src_len, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)),
                                 tgt_out.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(src, src_len, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)),
                             tgt_out.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        n_tok = (tgt_out != PAD).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
    mean = total_loss / max(1, total_tokens)
    return mean, math.exp(min(20.0, mean))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--limit", type=int, default=0, help="cap training pairs")
    ap.add_argument("--epochs", type=int, default=0, help="override cfg epochs")
    ap.add_argument("--overfit", type=int, default=0,
                    help="train and validate on the same N pairs (sanity check)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    ensure_dirs()
    set_seed(cfg["seed"])
    device = get_device()
    d, tr, tk = cfg["data"], cfg["train"], cfg["tokenizer"]
    epochs = args.epochs or tr["epochs"]

    sp = spm.SentencePieceProcessor(model_file=tk["model_file"])
    vocab_size = sp.get_piece_size()

    train_ds = QGDataset(d["train_tsv"], sp, d["max_src_pieces"], d["max_tgt_pieces"],
                         limit=args.overfit or args.limit)
    valid_ds = (train_ds if args.overfit else
                QGDataset(d["valid_tsv"], sp, d["max_src_pieces"], d["max_tgt_pieces"]))
    print(f"train pairs: {len(train_ds)}   valid pairs: {len(valid_ds)}")

    bs = min(tr["batch_size"], len(train_ds))
    train_loader = make_loader(train_ds, bs, True, tr["bucket_multiplier"])
    valid_loader = make_loader(valid_ds, bs, False)

    model = build_model(cfg, vocab_size).to(device)
    n_params = count_params(model)
    print(f"trainable parameters: {n_params:,}   ->  Table 2")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD,
                                    label_smoothing=tr.get("label_smoothing", 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=tr["lr"],
                                 weight_decay=tr["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=tr["lr_factor"], patience=tr["lr_patience"])
    scaler = (torch.cuda.amp.GradScaler()
              if tr["amp"] and device.type == "cuda" else None)

    log_path = tr["log_csv"]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_ppl", "valid_loss", "valid_ppl", "lr", "secs"])

    best, bad_epochs, t_total = float("inf"), 0, time.time()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tl, tppl = train_one_epoch(model, train_loader, optimizer, criterion,
                                   device, tr["clip"], scaler)
        vl, vppl = evaluate_loss(model, valid_loader, criterion, device)
        scheduler.step(vl)
        lr_now = optimizer.param_groups[0]["lr"]
        secs = time.time() - t0
        print(f"epoch {ep:02d} | train {tl:.4f} (ppl {tppl:.1f}) | "
              f"valid {vl:.4f} (ppl {vppl:.1f}) | lr {lr_now:.2e} | {secs:.0f}s")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, round(tl, 5), round(tppl, 3), round(vl, 5),
                                    round(vppl, 3), lr_now, round(secs, 1)])

        if vl < best - 1e-4:
            best, bad_epochs = vl, 0
            save_checkpoint(tr["ckpt_path"], model, optimizer, ep, vl, cfg, vocab_size)
            print(f"  saved best checkpoint -> {tr['ckpt_path']}")
        else:
            bad_epochs += 1
            if bad_epochs >= tr["early_stop_patience"]:
                print("early stopping")
                break

    wall = (time.time() - t_total) / 60
    print(f"\nbest valid loss {best:.4f} (ppl {math.exp(min(20, best)):.2f})")
    print(f"wall clock {wall:.1f} min on {torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}")
    print(f"parameters {n_params:,}  |  vocab {vocab_size}  ->  Table 2")


if __name__ == "__main__":
    main()
