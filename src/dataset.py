"""Dataset, padding collate, and a length-bucketing sampler."""
import csv
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.common import BOS, EOS, PAD


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        return [(row[0], row[1]) for row in r if len(row) >= 2]


class QGDataset(Dataset):
    """Encodes (source, target) text pairs into id sequences once, up front."""

    def __init__(self, path, sp, max_src=128, max_tgt=48, limit=0):
        self.pairs = read_tsv(path)
        if limit:
            self.pairs = self.pairs[:limit]
        self.data = []
        for src, tgt in self.pairs:
            s = sp.encode(src)[:max_src]
            t = sp.encode(tgt)[:max_tgt]
            if not s or not t:
                continue
            self.data.append((s, t))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        s, t = self.data[i]
        return torch.tensor(s), torch.tensor([BOS] + t), torch.tensor(t + [EOS])

    def src_lengths(self):
        return [len(s) for s, _ in self.data]


def collate(batch):
    srcs, tgt_ins, tgt_outs = zip(*batch)
    src_len = torch.tensor([len(s) for s in srcs], dtype=torch.long)
    S = max(len(s) for s in srcs)
    T = max(len(t) for t in tgt_ins)
    B = len(batch)

    src = torch.full((B, S), PAD, dtype=torch.long)
    tgt_in = torch.full((B, T), PAD, dtype=torch.long)
    tgt_out = torch.full((B, T), PAD, dtype=torch.long)
    for i, (s, ti, to) in enumerate(zip(srcs, tgt_ins, tgt_outs)):
        src[i, :len(s)] = s
        tgt_in[i, :len(ti)] = ti
        tgt_out[i, :len(to)] = to
    return src, src_len, tgt_in, tgt_out


class BucketBatchSampler(Sampler):
    """Shuffle, then sort within large pools so batches have similar lengths.

    Cuts padded compute substantially versus pure random batching while keeping
    enough randomness that batch composition still changes every epoch.
    """

    def __init__(self, lengths, batch_size, multiplier=50, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.pool = batch_size * multiplier
        self.shuffle = shuffle

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(idx)
        batches = []
        for i in range(0, len(idx), self.pool):
            pool = sorted(idx[i:i + self.pool], key=lambda j: self.lengths[j])
            for k in range(0, len(pool), self.batch_size):
                batches.append(pool[k:k + self.batch_size])
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

DEFAULT_WORKERS = 0 if os.name == "nt" else 2
def make_loader(ds, batch_size, shuffle=True, bucket_multiplier=50, num_workers=DEFAULT_WORKERS):
    if shuffle:
        sampler = BucketBatchSampler(ds.src_lengths(), batch_size, bucket_multiplier, True)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                          num_workers=num_workers)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                      num_workers=num_workers)
