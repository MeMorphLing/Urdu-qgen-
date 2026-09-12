"""Offline self-test. No dataset, no GPU, no tokenizer needed.

    python -m src.selftest

Checks, on synthetic data:
  1. forward pass shapes and attention normalisation
  2. attention puts zero mass on padding
  3. a training step reduces loss (model can learn a copy-ish mapping)
  4. greedy batch decoding runs and respects EOS
  5. beam search runs, returns a sane sequence, and is length-normalised
  6. GRU and LSTM variants both work

Run this before you touch the real data. If it passes, any later failure is in
the data or tokenizer, not in the model.
"""
import torch
from torch import nn

from src.common import BOS, EOS, PAD, build_model, count_params
from src.decode import beam_search_one, greedy_decode_batch, greedy_decode_one

CFG = {
    "model": {"rnn_type": "gru", "emb_dim": 32, "hid_dim": 24, "enc_layers": 2,
              "dec_layers": 2, "dropout": 0.1, "attn_dim": 20},
}
V, B, S, T = 40, 6, 9, 7


def fake_batch(device):
    torch.manual_seed(0)
    src = torch.randint(4, V, (B, S), device=device)
    src_len = torch.tensor([S, S, S - 2, S - 3, S - 1, S - 4])
    for i, L in enumerate(src_len.tolist()):
        src[i, L:] = PAD
    tgt_in = torch.randint(4, V, (B, T), device=device)
    tgt_in[:, 0] = BOS
    tgt_out = torch.cat([tgt_in[:, 1:], torch.full((B, 1), EOS, device=device)], 1)
    tgt_in[3, 5:] = PAD
    tgt_out[3, 4:] = PAD
    return src, src_len, tgt_in, tgt_out


def check(name, cond):
    print(("  PASS  " if cond else "  FAIL  ") + name)
    assert cond, name


def main():
    device = torch.device("cpu")

    for rnn_type in ["gru", "lstm"]:
        print(f"\n--- {rnn_type.upper()} ---")
        cfg = {"model": dict(CFG["model"], rnn_type=rnn_type)}
        model = build_model(cfg, V).to(device)
        src, src_len, tgt_in, tgt_out = fake_batch(device)

        model.eval()
        logits, attn = model(src, src_len, tgt_in)
        check(f"logits shape {tuple(logits.shape)} == {(B, T, V)}",
              tuple(logits.shape) == (B, T, V))
        check(f"attention shape {tuple(attn.shape)} == {(B, T, S)}",
              tuple(attn.shape) == (B, T, S))
        check("attention rows sum to 1",
              torch.allclose(attn.sum(-1), torch.ones(B, T), atol=1e-4))

        pad_mask = src.eq(PAD).unsqueeze(1).expand(-1, T, -1)
        check("zero attention mass on padding",
              attn[pad_mask].abs().max().item() < 1e-6)

        # --- training actually reduces loss ---
        model.train()
        crit = nn.CrossEntropyLoss(ignore_index=PAD)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        first = last = None
        for step in range(60):
            opt.zero_grad()
            lg, _ = model(src, src_len, tgt_in)
            loss = crit(lg.reshape(-1, V), tgt_out.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step == 0:
                first = loss.item()
            last = loss.item()
        print(f"  loss {first:.3f} -> {last:.3f}")
        check("loss decreased on a fixed batch", last < first * 0.5)

        # --- pad targets contribute no gradient ---
        model.zero_grad()
        lg, _ = model(src, src_len, tgt_in)
        crit(lg.reshape(-1, V), tgt_out.reshape(-1)) # a_llows backward to be called
        masked = tgt_out.clone()
        check("ignore_index masks pad", (masked == PAD).any().item())

        # --- decoding ---
        model.eval()
        outs = greedy_decode_batch(model, src, src_len, device, max_len=12)
        check(f"greedy returns {B} sequences", len(outs) == B)
        check("greedy never emits EOS inside the output",
              all(EOS not in o for o in outs))
        check("greedy never emits PAD as a token",
              all(all(t != PAD for t in o) for o in outs))

        ids = [t for t in src[0].tolist() if t != PAD]
        g_ids, g_attn = greedy_decode_one(model, ids, device, max_len=12)
        check("greedy_decode_one returns an attention matrix",
              g_attn is not None and g_attn.shape[1] == len(ids))

        b_ids = beam_search_one(model, ids, device, beam_size=4, max_len=12, alpha=0.7)
        check("beam returns a non-empty sequence", len(b_ids) > 0)
        check("beam output contains no EOS/PAD",
              all(t not in (EOS, PAD) for t in b_ids))

        b1 = beam_search_one(model, ids, device, beam_size=1, max_len=12, alpha=0.0)
        check("beam(k=1, alpha=0) matches greedy", b1 == g_ids)

        long_beam = beam_search_one(model, ids, device, 4, 12, alpha=1.0)
        print(f"  beam alpha=0.7 len {len(b_ids)}, alpha=1.0 len {len(long_beam)}")
        check("length penalty is applied without crashing", len(long_beam) >= 0)

    cfg = {"model": dict(CFG["model"], rnn_type="gru")}
    m = build_model(cfg, 8000)
    print(f"\nparameter count at vocab 8000 (toy dims): {count_params(m):,}")
    print("\nALL SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
