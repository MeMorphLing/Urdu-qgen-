# Urdu Answer-Aware Question Generation

Given an Urdu sentence with the answer marked as `<ans>...</ans>`, generate the
question that the answer responds to. RNN encoder–decoder with Bahdanau
attention, trained from scratch on UQA. No pretrained weights, no Transformers.

```
دریائے سندھ لگ بھگ <ans> 3180 کلومیٹر </ans> لمبا ہے۔
                    ↓
        دریائے سندھ کتنا لمبا ہے؟
```

**Final model:** BLEU-4 5.36 (beam) / 5.35 (greedy) on UQA validation,
ROUGE-L 0.2542, perplexity 29.86, `<unk>` rate 0.000.

---

## Requirement → where it lives

| Spec item | File | Output |
|---|---|---|
| 1.1 load UQA, keep answerable | `src/prepare_data.py` | — |
| 1.2 locate answer sentence by offset | `split_sentences`, `make_pair` | — |
| 1.3 wrap answer, build src/tgt | `make_pair` | — |
| 1.4 length filter (60 / 25 words) | `configs/base.yaml` → `max_src_words` | — |
| 1.5 TSVs, counts, histograms | `prepare_data.py`, `src/viz.py --what hist` | `data/*.tsv`, `results/dataset_stats.json`, `results/figures/length_hist_*.png` |
| 2.1 SentencePiece 8k | `src/train_tokenizer.py` | `tokenizer/ur_sp.model` |
| 2.2 user-defined + reserved symbols | same, with assertions | — |
| 2.3 five tokenised examples | stdout of `train_tokenizer.py` | see §2.3 below |
| 3 model from primitives | `src/model.py` | — |
| 3 training, checkpointing, logging | `src/train.py` | `checkpoints/best.pt`, `results/train_log.csv` |
| 3 greedy + beam | `src/decode.py` | — |
| 4 metrics on both splits | `src/evaluate.py` | `results/metrics.json`, `results/samples.tsv` |
| 4.5 figures | `src/viz.py` | `results/figures/` |
| 4.7 discussion evidence | `src/analyze.py` | `results/analysis.json` |
| 3.2 human eval + κ | `src/human_eval.py` | `results/human_eval_*.csv` |
| 5 front end | `app/streamlit_app.py` | `results/figures/frontend.png` |
| — | `src/make_tables.py` | `results/tables.md` |
| — | `src/selftest.py`, `tools/integration_test.py` | offline correctness checks |

---

## Setup

```bash
pip install -r requirements.txt
python -m src.selftest         # verifies the model offline in ~30s
bash run.sh                    # or run the steps below individually
```

## Tests

Both run without downloading UQA, so you can check the code works before
spending GPU time.

```bash
python -m src.selftest            # shapes, attention masking, greedy, beam
python -m tools.integration_test  # synthetic corpus through the whole pipeline
```

`selftest` asserts that attention rows sum to 1, that padding receives exactly
zero attention mass, that a training step reduces loss, and that
`beam(k=1, alpha=0)` reproduces greedy — the cheapest proof that the beam
implementation is not silently broken.

## Pipeline

```bash
python -m src.prepare_data --config configs/base.yaml --stream
python -m src.train_tokenizer --config configs/base.yaml

# sanity ladder — do not skip these
python -m src.train --overfit 32 --epochs 30     # loss must approach 0
python -m src.train --limit 10000 --epochs 1     # loss must fall visibly

python -m src.train --config configs/base.yaml   # ~80 min on a T4
python -m src.evaluate --config configs/base.yaml
python -m src.analyze
python -m src.viz --what all --index 0
python -m src.make_tables

python -m src.human_eval --make                  # rate independently, then:
python -m src.human_eval --score

streamlit run app/streamlit_app.py
```

Training was run on Kaggle (Tesla T4); see `notebooks/kaggle_training.ipynb`.
`--stream` reads UQA over HTTP without caching the dataset to disk.

---

## Architecture

```
src  [B,S] ──► Embedding(8000,256)
             └► 2-layer bidirectional GRU(512)  ──► enc_out [B,S,1024]
                                                  └► final states [2,B,1024]
                                                     │ Bridge: tanh(Linear(1024→512))
                                                     ▼
                                              dec hidden [2,B,512]

per step t:
  query   = hidden[-1]                                    [B,512]
  score   = v^T tanh(W_enc·enc_out + W_dec·query)         [B,S]   ← masked at PAD
  alpha   = softmax(score)
  context = alpha · enc_out                               [B,1024]
  out,h   = GRU([emb(y_{t-1}) ; context])                 [B,512]
  logits  = Linear([out ; context ; emb(y_{t-1})])        [B,8000]
```

Attention is **strictly Bahdanau**: the context comes from the *previous*
decoder state and is fed *into* the RNN. Luong would compute it from the current
post-RNN state instead.

---

## Table 1 — Dataset statistics

| | Train | Validation | Wiki-UQA |
|---|---|---|---|
| Rows in raw dataset | 124,745 | 16,824 | 210 |
| Answerable rows | 83,018 | 11,169 | 210 |
| Pairs after length filter | 74,379 | 8,212 | 178 |
| Mean source / target length (words) | 32.33 / 11.93 | 32.94 / 12.32 | 31.61 / 11.41 |

**Drop reasons, train split:** 41,727 unanswerable (SQuAD 2.0 has no answer for
these), 5,967 source over 60 words, 1,584 offset mismatch, 1,020 target over 25
words, 68 exact duplicates. Of the answerable rows, 89.6% survived the filters.
`drop_no_sentence_found` was **zero** on all three splits — the delimiter set
located the answer's sentence every time.

Two observations worth noting. The validation split lost **1,767** rows to
deduplication against only 68 in train: SQuAD's dev set carries multiple
reference answers per question, and UQA flattened those into separate rows.
Without the dedup, validation metrics would be over-weighted toward whichever
questions happened to receive three annotators. Separately, the mandated 60-word
source cap costs 5,967 training pairs — 7% of the usable data — which is a known
constraint on how much signal the model ever sees.

## Table 2 — Model configuration

| | |
|---|---|
| Encoder / decoder type | 2-layer BiGRU / 2-layer GRU |
| Attention | Bahdanau (additive), masked at PAD |
| Embedding / hidden / attn dim | 256 / 512 / 512 |
| Dropout | **0.4** (see *Deviations from spec* below) |
| Vocabulary | 8,000 (shared, unigram SentencePiece) |
| Trainable parameters | 31,173,440 |
| Optimiser | Adam, lr 1e-3, ReduceLROnPlateau (×0.5, patience 2), label smoothing 0.1, grad clip 1.0 |
| Batch / epochs / wall-clock / GPU | 64 / 30 (best at epoch 29) / 1 h 20 min / Tesla T4 |

## Table 3 — Automatic metrics

| Split | Decoding | BLEU-4 | ROUGE-L | PPL | `<unk>` % |
|---|---|---|---|---|---|
| UQA valid | greedy | 5.35 | 0.2451 | 29.86 | 0.000 |
| UQA valid | beam (k=5) | 5.36 | 0.2542 | 29.86 | 0.000 |
| Wiki-UQA | greedy | 3.53 | 0.2197 | 39.71 | 0.000 |
| Wiki-UQA | beam (k=5) | 4.09 | 0.2312 | 39.71 | 0.000 |

BLEU is sacrebleu corpus-level on detokenised output with the `13a` tokenizer;
`BLEU-4_intl` is also in `metrics.json` as a cross-check. Perplexity is
token-level `exp(mean CE)` over non-pad targets from the best checkpoint. Beam
was run over a fixed 2,000-example subset of UQA validation and all 178
Wiki-UQA pairs; greedy over both splits in full.

**BLEU 5.36 sits below the 6–13 sanity band.** The model is converged, not
buggy — see *Training progression* and *§4.7* for the evidence and the reading.

## Table 4 — Human evaluation (50 samples, seed 1337)

*Fill from `results/human_eval_summary.csv` after both members rate
independently and `python -m src.human_eval --score` is run.*

| | Fluency | Relevance | Answerability |
|---|---|---|---|
| Member 1 (% yes) | | | |
| Member 2 (% yes) | | | |
| Cohen's κ | | | |

---

## Training progression

Three runs. Each configuration change was diagnosed from the validation loss
curve rather than tuned blindly.

| Run | Change | Best epoch | BLEU-4 greedy / beam | PPL |
|---|---|---|---|---|
| 1 | spec baseline: dropout 0.3, 15 epochs, no smoothing | 3 of 6 (early stop) | 3.58 / 4.16 | 34.97 |
| 2 | + dropout 0.4, label smoothing 0.1, patience 6 | 15 of 15 | 4.99 / 5.24 | 30.85 |
| 3 | + epoch cap 30 | 29 of 30 | **5.35 / 5.36** | **29.86** |

Run 1 overfit sharply: train loss fell 4.23 → 2.13 while validation bottomed at
epoch 3 and then climbed, triggering early stopping at epoch 6. Regularisation
moved the validation floor from epoch 3 to epoch 15, at which point the model
was still improving when it hit the epoch cap — so run 3 raised only the cap and
changed nothing else. It converged at epoch 29 with the curve flat over the final
ten epochs (4.3733 → 4.3570).

Note that runs 2 and 3 report *higher* absolute loss than run 1 despite being
better models: label smoothing adds a constant entropy term, so cross-entropy
and perplexity are not comparable across the smoothing boundary. BLEU and
ROUGE-L are.

One incidental finding: at epoch 7 the LR halved after a plateau and epoch 8
immediately set a new best. `ReduceLROnPlateau` earned its place — the smaller
step size found ground the larger one was stepping over.

## Deviations from spec

- **Dropout 0.4, not the specified 0.3.** Run 1 at the specified value overfit
  by epoch 3 (see above). The increase, together with label smoothing, moved the
  validation floor by twelve epochs and raised BLEU from 4.16 to 5.24. Declared
  here rather than left for a reader to notice.
- **`lr_patience` 2 and `early_stop_patience` 6**, raised from 1 and 3, for the
  same reason.
- Everything the brief pins — embedding 256, hidden 512, 2 layers each side,
  vocab 8k, the 60/25-word filters, GRU cell, from-scratch training — is
  unchanged.

---

## §2.3 Tokenizer notes

*Paste the five tokenised examples from `python -m src.train_tokenizer` here and
comment on: how Urdu postpositions (کا / کے / کی / میں / سے) separate into their
own pieces; where compound verbs such as حاصل کی split; how rare proper nouns
fragment toward characters while frequent words stay whole.*

Alphabet size 758, character coverage 1.0, 8,000 pieces, corpus 148,758 lines
(train sources + targets only — never validation, never Wiki-UQA).

## §4.6 Qualitative samples

*Pick ten from `results/samples.tsv` — five good, five bad. Failure types
observed in this model's output:*

- **Entity substitution** — ڈاربی → ڈارون (Darby → Darwin): phonetically
  adjacent, semantically wrong.
- **Hallucinated entity** — کینے ویسٹ, الزبتھ and "10 نومبر 2008" appear in
  generated questions but nowhere in the source sentence.
- **Repetition** — قومی تاریخی طور پر قومی تاریخی نشان
- **Wrong question word** — کہاں generated where the reference uses کس.
- **Near-miss penalised by the metric** — reference
  لاس اینجلس کے علاقے میں کتنے باشندے ہیں؟ vs beam
  لاس اینجلس میں کتنے لوگ رہتے ہیں؟. Both are correct questions; the beam
  output scores 44 sentence-BLEU points lower for dropping one word.

Degenerate behaviour is rare: copied source 0.1%, repetition 1.1%, empty 0.0%,
`<unk>` 0.000%.

## §4.7 Discussion

### Which question words does the model get right?

| Question word | n | accuracy | mean sentBLEU |
|---|---|---|---|
| کتنے (how many) | 88 | 67.0% | 10.58 |
| کتنا (how much) | 47 | 61.7% | 12.41 |
| کیا (what) | 586 | 53.2% | 6.55 |
| کس (which/whom) | 650 | 43.8% | 7.08 |
| کب (when) | 123 | 41.5% | 9.36 |
| کون (who) | 316 | 31.0% | 8.28 |
| کہاں (where) | 56 | 23.2% | 8.94 |
| کتنی (how much, fem.) | 37 | 21.6% | 10.95 |
| کیوں (why) | 37 | 2.7% | 5.84 |
| کیسے (how) | 13 | 0.0% | 3.83 |

The ranking tracks how strongly the answer span's *surface form* constrains the
question type. A numeric span forces کتنے / کتنا. A date suggests کب. A person's
name suggests کون — but only 31%, because a name can answer "who wrote X",
"who was defeated", or "whose theory". And کیوں / کیسے sit at 2.7% and 0.0%:
cause and manner are not recoverable from a marked span at all, so the model has
no signal to condition on. The span tells it *what kind of thing* the answer is,
never *why* it is the answer.

### Where does beam search help, and where does it hurt?

Barely either, at convergence. Beam wins 835 cases, loses 717, ties 448 — a mean
sentence-BLEU delta of +0.68 that nets out to +0.01 corpus BLEU. In run 1 beam
gained 0.58 BLEU over greedy; by run 3 the gap is 0.01. **The greedy-to-beam gap
is itself a measure of model uncertainty**: a converged model concentrates
probability mass on one path, so widening the search stops finding anything new.

Where beam does hurt, the mechanism is visible. Beam produces shorter output
(10.6 words vs greedy's 12.21, against references averaging 11.93) and, when the
model is confidently wrong, searches out a *more fluent* wrong question. The
worst case in `analysis.json` has beam inventing "کینے ویسٹ" — an entity absent
from the source — while greedy stayed closer with a clumsier near-miss.

### Why does performance drop on Wiki-UQA?

Corpus BLEU falls 5.36 → 4.09 and perplexity rises 29.86 → 39.71. The standard
reading is distribution shift: UQA is machine-translated SQuAD, so the model
learns translationese and SQuAD's question-template distribution, while Wiki-UQA
is natively Urdu.

That reading is right but incomplete, and the per-question-word breakdown shows
why. Accuracy on کس actually *rises* out of domain (43.8% → 59.4%) and کیا holds
(53.2% → 56.9%), while کب (41.5% → 20.0%) and کہاں (23.2% → 10.0%) collapse.
Wiki-UQA skews toward entity questions the model already handles and away from
temporal and locative ones it does not, so part of the aggregate drop is
**composition, not domain shift**. With n = 15 and n = 10 for those categories
the per-word figures are noisy, and Wiki-UQA is only 178 pairs in total, so the
corpus BLEU gap should not be over-read either.

### On the metric itself

BLEU-4 understates this model's output quality, and the LA example above
demonstrates it concretely: two grammatical, answerable, near-synonymous
questions separated by 44 sentence-BLEU points because one word differs. On
~11-token sentences, a single 4-gram miss is catastrophic to the score. The human
evaluation (Table 4) exists precisely to measure what BLEU cannot.

---

## Figures

- `results/figures/loss_curve.png` — training and validation loss per epoch
- `results/figures/length_hist_train.png`, `length_hist_valid.png`
- `results/figures/attention_0.png` — source tokens × generated tokens
- `results/figures/frontend.png` — front-end screenshot

---

## Known traps (all handled in this repo)

1. **The hub dataset schema is not the one in the assignment appendix.**
   `uqa/UQA` ships **flat** columns — `answer` (str), `answer_start` (int),
   `is_impossible` (bool) — not the nested
   `answers={"text": [...], "answer_start": [...]}` the starter code assumes.
   The unmodified appendix code yields **zero** usable pairs. `normalize_answer`
   in `prepare_data.py` accepts flat, nested and list-of-dicts shapes.
2. **`rouge_score` returns 0.0 on Urdu.** Its default tokenizer runs
   `re.sub(r"[^a-z0-9]+", " ", …)`, deleting every Urdu character. Measured on
   one Urdu pair: default tokenizer 0.0, whitespace tokenizer 0.857.
   `src/evaluate.py` passes a whitespace tokenizer.
3. **Beam without length normalisation loses to greedy.** Each extra token
   subtracts log-probability, so an unnormalised beam converges on 3-token
   questions. Scores are divided by `len**0.7`.
4. **Unmasked attention** would put probability mass on padding, in an amount
   depending on batch composition. `BahdanauAttention` masks before softmax;
   `selftest.py` asserts the mass is exactly zero.
5. **Round-trip equality is the wrong tokenizer invariant.** `nmt_nfkc`
   normalisation is lossy by design — it decomposes U+2047 (which UQA itself
   contains, left over from its translation pipeline) into `??`. The correct
   check is **idempotence**: `decode(encode(x))` must be stable under a second
   pass. Incidentally this keeps the `<unk>` metric honest, since a decoded
   `<unk>` surfaces as U+2047 while a literal U+2047 in the text normalises to
   `??` — the two cannot be confused.
6. **`<ans>` atomicity.** Encoding a *bare* tag returns `['▁', '<ans>']` because
   of `add_dummy_prefix` — that is normal. What matters, and what is asserted,
   is that the tag survives as exactly one piece inside a real sentence.
7. **Matplotlib renders Urdu as boxes.** `src/viz.py` uses `arabic_reshaper` +
   `python-bidi` and falls back to indexed labels with a printed legend if no
   Urdu font is present. Drop a `.ttf` into `fonts/`.
8. **Windows.** DataLoader workers are spawned, not forked, so the dataset must
   pickle — a `SentencePieceProcessor` is a SWIG object and cannot, which hangs
   silently. `DEFAULT_WORKERS` is 0 on `nt`. Consoles also default to cp1252 and
   die on Urdu output; `configure_stdout()` in `src/common.py` forces UTF-8.
9. **Tokenizer leakage.** The SentencePiece corpus is built from the train split
   only — never validation, never Wiki-UQA.

---

## Viva notes

Both members must be able to answer all of these.

- **Why bidirectional encoder, unidirectional decoder?** The source is fully
  available, so both directions of context are legal. The decoder generates
  left-to-right at inference, so it cannot see its own future.
- **What is the bridge for?** The encoder's final state is 1024-d (two
  directions concatenated); the decoder is 512-d. `tanh(Linear(1024→512))` per
  layer maps between them.
- **Draw the Bahdanau score.** `v^T tanh(W_enc·h_s + W_dec·h_{t-1})`, with
  `W_enc: 1024→512`, `W_dec: 512→512`, `v: 512→1`.
- **Why mask before softmax?** Padding would otherwise receive probability mass,
  in an amount depending on batch composition.
- **What does `ignore_index=0` do?** Pad positions contribute no loss and no
  gradient, so the model is not rewarded for predicting pad.
- **Why is PPL teacher-forced but BLEU free-running?** PPL measures the density
  the model assigns to the reference given gold prefixes. BLEU measures what the
  model produces from its own prefixes — which is why exposure bias shows up in
  BLEU and not in PPL.
- **Why is `model.py` not the same as `decode.py`?** `Decoder` is the network
  that computes one step. `decode.py` holds the search algorithms that call that
  step repeatedly and choose which token to take. Same weights, different search.
- **What is early stopping doing?** Halting when validation stops improving, and
  keeping the best checkpoint rather than the last. In run 1 train loss kept
  falling for three epochs after validation turned — the saved model is from the
  validation floor, not from where training stopped.
- **Why did BLEU come in below the sanity band?** Converged, not broken: see
  *Training progression*. The contributing factors are a constrained
  architecture, 74k pairs of machine-translated training data, and a metric that
  punishes near-synonymous phrasing on short sentences.
- **Where does beam help / hurt, and which question words work?** See §4.7 —
  answer with the numbers, not the intuition.
