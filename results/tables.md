## Table 1 — Dataset statistics

|  | Train | Validation | Wiki-UQA |
|---|---|---|---|
| Rows in raw dataset | 124,745 | 16,824 | 210 |
| Answerable rows | 83,018 | 11,169 | 210 |
| Pairs after length filter | 74,379 | 8,212 | 178 |
| Mean source / target length (words) | 32.33 / 11.93 | 32.94 / 12.32 | 31.61 / 11.41 |

## Table 2 — Model configuration

*`checkpoints/best.pt` not yet generated — run `python -m src.train --config configs/base.yaml`*

*`results/train_log.csv` not yet generated — run `python -m src.train --config configs/base.yaml`*

## Table 3 — Automatic metrics

*`results/metrics.json` not yet generated — run `python -m src.evaluate --config configs/base.yaml`*

## Table 4 — Human evaluation

*`results/human_eval_summary.csv` not yet generated — run `python -m src.human_eval --score`*
