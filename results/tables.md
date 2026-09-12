## Table 1 — Dataset statistics

|  | Train | Validation | Wiki-UQA |
|---|---|---|---|
| Rows in raw dataset | 124,745 | 16,824 | 210 |
| Answerable rows | 83,018 | 11,169 | 210 |
| Pairs after length filter | 74,379 | 8,212 | 178 |
| Mean source / target length (words) | 32.33 / 11.93 | 32.94 / 12.32 | 31.61 / 11.41 |

## Table 2 — Model configuration

|  |  |
|---|---|
| Encoder / decoder type | 2-layer BiGRU / 2-layer GRU |
| Attention | Bahdanau (additive), masked at PAD |
| Embedding dim | 256 |
| Hidden dim | 512 |
| Attention dim | 512 |
| Dropout | 0.4 |
| Vocabulary | 8,000 (shared, unigram SentencePiece) |
| Trainable parameters | 31,173,440 |
| Optimiser | Adam, lr 0.001, ReduceLROnPlateau (x0.5, patience 2) |
| Batch size | 64 |
| Epochs actually run | 30 |
| Total wall-clock | 1 h 20 min |

## Table 3 — Automatic metrics

| Split | Decoding | BLEU-4 | ROUGE-L | PPL | `<unk>` % |
|---|---|---|---|---|---|
| UQA valid | greedy | 5.35 | 0.2451 | 29.86 | 0.000 |
| UQA valid | beam (k=5) | 5.36 | 0.2542 | 29.86 | 0.000 |
| Wiki-UQA | greedy | 3.53 | 0.2197 | 39.71 | 0.000 |
| Wiki-UQA | beam (k=5) | 4.09 | 0.2312 | 39.71 | 0.000 |

## Table 4 — Human evaluation

*`results/human_eval_summary.csv` not yet generated — run `python -m src.human_eval --score`*
