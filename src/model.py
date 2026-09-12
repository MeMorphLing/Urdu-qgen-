"""Task 3 - RNN encoder-decoder with Bahdanau attention, built from primitives.

Shapes (B=batch, S=source len, T=target len, E=emb, H=hidden, V=vocab):
    src        [B, S]
    enc_out    [B, S, 2H]        bidirectional encoder
    dec hidden [L, B, H]         bridged from the encoder's final states
    context    [B, 2H]           attention-weighted sum of enc_out
    logits     [B, T, V]

Attention is strictly Bahdanau (additive): the context is computed from the
PREVIOUS decoder state and concatenated with the embedding before entering the
decoder RNN, rather than after it (which would be Luong).
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, layers, dropout,
                 rnn_type="gru", pad_id=0):
        super().__init__()
        self.rnn_type = rnn_type
        self.layers = layers
        self.hid_dim = hid_dim
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(emb_dim, hid_dim, num_layers=layers, batch_first=True,
                           bidirectional=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)

    def _merge_directions(self, state):
        # [L*2, B, H] -> [L, B, 2H]
        _, B, H = state.shape # Unpacked variable `L2` is never used
        state = state.view(self.layers, 2, B, H).permute(0, 2, 1, 3)
        return state.reshape(self.layers, B, 2 * H)

    def forward(self, src, src_len):
        x = self.drop(self.emb(src))
        packed = pack_padded_sequence(x, src_len.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, state = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True,
                                     total_length=src.size(1))
        if self.rnn_type == "gru":
            state = self._merge_directions(state)
        else:
            h, c = state
            state = (self._merge_directions(h), self._merge_directions(c))
        return out, state


class Bridge(nn.Module):
    """Maps the encoder's final 2H states to the decoder's H states, per layer."""

    def __init__(self, hid_dim, enc_layers, dec_layers, rnn_type="gru"):
        super().__init__()
        self.rnn_type = rnn_type
        self.dec_layers = dec_layers
        self.enc_layers = enc_layers
        self.h_proj = nn.Linear(2 * hid_dim, hid_dim)
        self.c_proj = nn.Linear(2 * hid_dim, hid_dim) if rnn_type == "lstm" else None

    def _fit_layers(self, state):
        if self.dec_layers == self.enc_layers:
            return state
        if self.dec_layers < self.enc_layers:
            return state[-self.dec_layers:]
        pad = state[-1:].expand(self.dec_layers - self.enc_layers, -1, -1)
        return torch.cat([state, pad], dim=0)

    def forward(self, state):
        if self.rnn_type == "gru":
            return self._fit_layers(torch.tanh(self.h_proj(state)))
        h, c = state
        return (self._fit_layers(torch.tanh(self.h_proj(h))),
                self._fit_layers(torch.tanh(self.c_proj(c))))


class BahdanauAttention(nn.Module):
    """score(h_{t-1}, enc_s) = v^T tanh(W_enc * enc_s + W_dec * h_{t-1})"""

    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def project_keys(self, enc_out):
        """Precompute W_enc*enc_s once per sentence instead of once per step."""
        return self.W_enc(enc_out)

    def forward(self, query, enc_out, enc_keys, mask):
        # query [B, dec_dim]; enc_out [B, S, enc_dim]; mask [B, S] True = real
        scores = self.v(torch.tanh(enc_keys + self.W_dec(query).unsqueeze(1)))
        scores = scores.squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        alpha = F.softmax(scores, dim=-1)                    # [B, S]
        context = torch.bmm(alpha.unsqueeze(1), enc_out).squeeze(1)  # [B, 2H]
        return context, alpha


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, layers, dropout, attn_dim,
                 rnn_type="gru", pad_id=0):
        super().__init__()
        self.rnn_type = rnn_type
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.attn = BahdanauAttention(2 * hid_dim, hid_dim, attn_dim)
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(emb_dim + 2 * hid_dim, hid_dim, num_layers=layers,
                           batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        # deep output layer: decoder state + context + input embedding
        self.out = nn.Linear(hid_dim + 2 * hid_dim + emb_dim, vocab_size)

    @staticmethod
    def _top_h(hidden, rnn_type):
        return hidden[-1] if rnn_type == "gru" else hidden[0][-1]

    def step(self, y_prev, hidden, enc_out, enc_keys, mask):
        """One decoding step. y_prev [B] -> logits [B, V]."""
        e = self.drop(self.emb(y_prev))                       # [B, E]
        query = self._top_h(hidden, self.rnn_type)            # [B, H]
        context, alpha = self.attn(query, enc_out, enc_keys, mask)
        rnn_in = torch.cat([e, context], dim=-1).unsqueeze(1)
        out, hidden = self.rnn(rnn_in, hidden)
        top = self.drop(out.squeeze(1))                       # [B, H]
        logits = self.out(torch.cat([top, context, e], dim=-1))
        return logits, hidden, alpha


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hid_dim=512, enc_layers=2,
                 dec_layers=2, dropout=0.3, attn_dim=512, rnn_type="gru",
                 pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.rnn_type = rnn_type
        self.encoder = Encoder(vocab_size, emb_dim, hid_dim, enc_layers, dropout,
                               rnn_type, pad_id)
        self.bridge = Bridge(hid_dim, enc_layers, dec_layers, rnn_type)
        self.decoder = Decoder(vocab_size, emb_dim, hid_dim, dec_layers, dropout,
                               attn_dim, rnn_type, pad_id)

    def encode(self, src, src_len):
        enc_out, state = self.encoder(src, src_len)
        hidden = self.bridge(state)
        enc_keys = self.decoder.attn.project_keys(enc_out)
        mask = src.ne(self.pad_id)
        return enc_out, enc_keys, mask, hidden

    def forward(self, src, src_len, tgt_in):
        """Teacher-forced training pass. Returns (logits [B,T,V], attn [B,T,S])."""
        enc_out, enc_keys, mask, hidden = self.encode(src, src_len)
        logits, attns = [], []
        for t in range(tgt_in.size(1)):
            step_logits, hidden, alpha = self.decoder.step(
                tgt_in[:, t], hidden, enc_out, enc_keys, mask)
            logits.append(step_logits)
            attns.append(alpha)
        return torch.stack(logits, dim=1), torch.stack(attns, dim=1)
