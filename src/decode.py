"""Greedy and beam-search decoding.

Beam scores are length-normalised (score = sum_logprob / len**alpha). Without
this the beam systematically prefers very short questions, because every extra
token subtracts log-probability - the classic "beam is worse than greedy" bug.
"""
import torch
import torch.nn.functional as F

from src.common import BOS, EOS


def _reorder_hidden(hidden, index, rnn_type):
    if rnn_type == "gru":
        return hidden.index_select(1, index)
    return (hidden[0].index_select(1, index), hidden[1].index_select(1, index))


def _expand_hidden(hidden, k, rnn_type):
    if rnn_type == "gru":
        return hidden.repeat_interleave(k, dim=1)
    return (hidden[0].repeat_interleave(k, dim=1),
            hidden[1].repeat_interleave(k, dim=1))


@torch.no_grad()
def greedy_decode_batch(model, src, src_len, device, max_len=40):
    """Batched greedy decoding. Returns a list of id lists (EOS stripped)."""
    model.eval()
    src = src.to(device)
    enc_out, enc_keys, mask, hidden = model.encode(src, src_len)
    B = src.size(0)
    y = torch.full((B,), BOS, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    outputs = [[] for _ in range(B)]

    for _ in range(max_len):
        logits, hidden, _ = model.decoder.step(y, hidden, enc_out, enc_keys, mask)
        y = logits.argmax(-1)
        for i in range(B):
            if not done[i] and y[i].item() != EOS:
                outputs[i].append(y[i].item())
        done = done | (y == EOS)
        if bool(done.all()):
            break
    return outputs


@torch.no_grad()
def greedy_decode_one(model, src_ids, device, max_len=40):
    """Greedy decode a single sentence, also returning the attention matrix."""
    model.eval()
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_len = torch.tensor([len(src_ids)], dtype=torch.long)
    enc_out, enc_keys, mask, hidden = model.encode(src, src_len)
    y = torch.tensor([BOS], dtype=torch.long, device=device)
    out_ids, attns = [], []
    for _ in range(max_len):
        logits, hidden, alpha = model.decoder.step(y, hidden, enc_out, enc_keys, mask)
        y = logits.argmax(-1)
        tok = y.item()
        attns.append(alpha.squeeze(0).cpu())
        if tok == EOS:
            break
        out_ids.append(tok)
    attn = torch.stack(attns).numpy() if attns else None   # [T_out, S]
    return out_ids, attn


@torch.no_grad()
def beam_search_one(model, src_ids, device, beam_size=5, max_len=40, alpha=0.7):
    """Beam search over a single sentence. Returns the best id list."""
    model.eval()
    k = beam_size
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_len = torch.tensor([len(src_ids)], dtype=torch.long)
    enc_out, enc_keys, mask, hidden = model.encode(src, src_len)

    enc_out = enc_out.expand(k, -1, -1).contiguous()
    enc_keys = enc_keys.expand(k, -1, -1).contiguous()
    mask = mask.expand(k, -1).contiguous()
    hidden = _expand_hidden(hidden, k, model.rnn_type)

    y = torch.full((k,), BOS, dtype=torch.long, device=device)
    scores = torch.full((k,), float("-inf"), device=device)
    scores[0] = 0.0                          # only beam 0 is live at step 0
    seqs = [[] for _ in range(k)]
    finished = []

    for _ in range(max_len):
        logits, new_hidden, _ = model.decoder.step(y, hidden, enc_out, enc_keys, mask)
        logprobs = F.log_softmax(logits.float(), dim=-1)          # [k, V]
        cand = scores.unsqueeze(1) + logprobs                     # [k, V]
        V = cand.size(1)
        flat = cand.view(-1)
        top_scores, top_idx = flat.topk(k)
        beam_idx = torch.div(top_idx, V, rounding_mode="floor")
        token_idx = top_idx % V

        new_seqs, keep_beam, keep_tok, keep_score = [], [], [], []
        for s, b, t in zip(top_scores.tolist(), beam_idx.tolist(), token_idx.tolist()):
            seq = seqs[b] + [t]
            if t == EOS:
                norm = s / (max(1, len(seq)) ** alpha)
                finished.append((norm, seq[:-1]))
            else:
                new_seqs.append(seq)
                keep_beam.append(b)
                keep_tok.append(t)
                keep_score.append(s)

        if len(finished) >= k or not new_seqs:
            break

        while len(new_seqs) < k:             # pad the beam back to width k
            new_seqs.append(new_seqs[-1][:])
            keep_beam.append(keep_beam[-1])
            keep_tok.append(keep_tok[-1])
            keep_score.append(float("-inf"))

        seqs = new_seqs
        idx = torch.tensor(keep_beam, dtype=torch.long, device=device)
        hidden = _reorder_hidden(new_hidden, idx, model.rnn_type)
        y = torch.tensor(keep_tok, dtype=torch.long, device=device)
        scores = torch.tensor(keep_score, device=device)

    if not finished:                          # nothing hit EOS within max_len
        best = max(range(len(seqs)), key=lambda i: scores[i].item())
        return seqs[best]
    finished.sort(key=lambda x: -x[0])
    return finished[0][1]
