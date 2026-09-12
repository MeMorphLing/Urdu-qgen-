"""Shared utilities: config, seeding, model construction, checkpoint IO."""
import math
import os
import random
import sys

import numpy as np
import torch
import yaml

PAD, UNK, BOS, EOS = 0, 1, 2, 3
ANS_OPEN, ANS_CLOSE = "<ans>", "</ans>"

def configure_stdout():
    """Force UTF-8 on stdout/stderr.

    Windows consoles and pipes default to cp1252, which cannot represent Urdu:
    printing a single tokenised example kills the process with
    UnicodeEncodeError. Every module imports this one, so calling it here fixes
    the whole project in one place.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


configure_stdout()


def load_cfg(path="configs/base.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs():
    for d in ["data", "tokenizer", "checkpoints", "results", "results/figures"]:
        os.makedirs(d, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg, vocab_size):
    from src.model import Seq2Seq
    m = cfg["model"]
    return Seq2Seq(
        vocab_size=vocab_size,
        emb_dim=m["emb_dim"],
        hid_dim=m["hid_dim"],
        enc_layers=m["enc_layers"],
        dec_layers=m["dec_layers"],
        dropout=m["dropout"],
        attn_dim=m["attn_dim"],
        rnn_type=m["rnn_type"],
        pad_id=PAD,
    )


def save_checkpoint(path, model, optimizer, epoch, val_loss, cfg, vocab_size):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "val_loss": val_loss,
        "cfg": cfg,
        "vocab_size": vocab_size,
    }, path)


def load_checkpoint(path, device=None):
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg, ckpt["vocab_size"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    """Teacher-forced loss. Returns (mean loss per token, perplexity)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for src, src_len, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits, _ = model(src, src_len, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        n_tok = (tgt_out != PAD).sum().item()
        total_loss += loss.item() * n_tok      # criterion uses reduction='mean'
        total_tokens += n_tok
    mean = total_loss / max(1, total_tokens)
    return mean, math.exp(min(20.0, mean))
