# Urdu Answer-Aware Question Generation

Given an Urdu sentence with the answer marked as `<ans>...</ans>`, generate the
question that the answer responds to. RNN encoder–decoder with Bahdanau
attention, trained from scratch on UQA. No pretrained weights, no Transformers.

```
دریائے سندھ لگ بھگ <ans> 3180 کلومیٹر </ans> لمبا ہے۔
                    ↓
        دریائے سندھ کتنا لمبا ہے؟
```

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
| 2.3 five tokenised examples | stdout of `train_tokenizer.py` | paste into blog |
| 3 model from primitives | `src/model.py` | — |
| 3 training, checkpointing, logging | `src/train.py` | `checkpoints/best.pt`, `results/train_log.csv` |
| 3 greedy + beam | `src/decode.py` | — |
| 4 metrics on both splits | `src/evaluate.py` | `results/metrics.json`, `results/samples.tsv` |
| 4.5 figures | `src/viz.py` | `results/figures/` |
| 4.7 discussion evidence | `src/analyze.py` | `results/analysis.json` |
| 3.2 human eval + κ | `src/human_eval.py` | `results/human_eval_*.csv` |
| 5 front end | `app/streamlit_app.py` | screenshot below |
| — | `src/selftest.py`, `tools/integration_test.py` | offline correctness checks |

---

## Setup

```bash
pip install -r requirements.txt
python -m src.selftest         # verifies the model offline in ~30s
bash run.sh                    # or run the steps below individually
```

## Tests

Both of these run without downloading UQA, so you can check the code works
before spending GPU time.

```bash
python -m src.selftest            # shapes, attention masking, greedy, beam
python -m tools.integration_test  # synthetic corpus through the whole pipeline
```

`selftest` asserts that attention rows sum to 1, that padding receives exactly
zero attention mass, that a training step reduces loss, and that
`beam(k=1, alpha=0)` reproduces greedy — which is the cheapest proof that the
beam implementation is not silently broken.

## Pipeline

```bash
python -m src.prepare_data --config configs/base.yaml
python -m src.train_tokenizer --config configs/base.yaml

# sanity ladder — do not skip these
python -m src.train --overfit 32 --epochs 30     # loss must approach 0
python -m src.train --limit 10000 --epochs 1     # loss must fall visibly

python -m src.train --config configs/base.yaml   # full run, 1–2 h on a T4/P100
python -m src.evaluate --config configs/base.yaml
python -m src.analyze
python -m src.viz --what all --index 0

python -m src.human_eval --make                  # rate independently, then:
python -m src.human_eval --score

streamlit run app/streamlit_app.py
```

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

Fill from `results/dataset_stats.json`.

| | Train | Validation | Wiki-UQA |
|---|---|---|---|
| Rows in raw dataset | | | |
| Answerable rows | | | |
| Pairs after length filter | | | |
| Mean source / target length | | | |

Note the drop reasons breakdown in the JSON — `drop_offset_mismatch` and
`drop_src_too_long` are the two big ones and are worth a sentence in the blog.

## Table 2 — Model configuration

| | |
|---|---|
| Encoder / decoder type | 2-layer BiGRU / 2-layer GRU |
| Attention | Bahdanau (additive), masked |
| Embedding / hidden / attn dim | 256 / 512 / 512 |
| Dropout | 0.3 |
| Vocabulary | 8000 (shared, unigram SentencePiece) |
| Trainable parameters | *printed by `src/train.py`* |
| Optimiser | Adam, lr 1e-3, ReduceLROnPlateau (×0.5, patience 1) |
| Batch / epochs / wall-clock / GPU | 64 / 15 / *fill* / *fill* |

## Table 3 — Automatic metrics

Fill from `results/metrics.json`.

| Split | Decoding | BLEU-4 | ROUGE-L | PPL | `<unk>` % |
|---|---|---|---|---|---|
| UQA valid | greedy | | | | |
| UQA valid | beam (k=5) | | | | |
| Wiki-UQA | greedy | | | | |
| Wiki-UQA | beam (k=5) | | | | |

BLEU is sacrebleu corpus-level on detokenised output with the `13a` tokenizer;
`BLEU-4_intl` is also reported in the JSON as a cross-check. Perplexity is
token-level `exp(mean CE)` over non-pad targets from the best checkpoint.

**Sanity band:** 6–13 on UQA validation, lower on Wiki-UQA. Near 0 means a bug.
Above 30 means train/validation leakage — go audit the splits.

## Table 4 — Human evaluation (50 samples, seed 1337)

Fill from `results/human_eval_summary.csv`.

| | Fluency | Relevance | Answerability |
|---|---|---|---|
| Member 1 (% yes) | | | |
| Member 2 (% yes) | | | |
| Cohen's κ | | | |

## Figures

- `results/figures/loss_curve.png`
- `results/figures/length_hist_train.png`, `length_hist_valid.png`
- `results/figures/attention_0.png`
- Front-end screenshot: `results/figures/frontend.png`

---

## Known traps (already handled in this repo)

1. **`rouge_score` returns 0.0 on Urdu.** Its default tokenizer runs
   `re.sub(r"[^a-z0-9]+", " ", …)`, which deletes every Urdu character.
   `src/evaluate.py` passes a whitespace tokenizer instead.
2. **Beam without length normalisation loses to greedy.** Each extra token
   subtracts log-probability, so an unnormalised beam converges on 3-token
   questions. We divide by `len**0.7`.
3. **Unmasked attention.** Softmax over padding correlates with sentence length
   and caps BLEU around 3. `BahdanauAttention` masks before softmax.
4. **`<ans>` not registered atomically.** If the user-defined symbol string
   doesn't match the string in the data exactly, SentencePiece shreds the tag.
   `train_tokenizer.py` asserts `sp.encode("<ans>") == ["<ans>"]` before anything
   downstream runs.
5. **Matplotlib renders Urdu as boxes.** `src/viz.py` uses `arabic_reshaper` +
   `python-bidi` and falls back to indexed labels with a printed legend if no
   Urdu font is installed. Drop a `.ttf` into `fonts/`.
6. **Tokenizer leakage.** The SentencePiece corpus is built from the train split
   only — never valid, never Wiki-UQA.

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
- **Why mask before softmax?** Padding positions would otherwise receive
  probability mass, and the amount would depend on batch composition.
- **What does `ignore_index=0` do?** Pad positions contribute no loss and no
  gradient, so the model isn't rewarded for predicting pad.
- **Why is PPL teacher-forced but BLEU free-running?** PPL measures the density
  the model assigns to the reference given gold prefixes. BLEU measures what the
  model actually produces from its own prefixes — which is why exposure bias
  shows up in BLEU and not in PPL.
- **Where does beam help / hurt?** It helps on longer questions where a locally
  suboptimal token opens a better continuation. It hurts when the model is
  confidently wrong: beam then finds a *more fluent* wrong question, which can
  score worse against the reference than greedy's clumsier near-miss. Numbers in
  `results/analysis.json`.
- **Which question words does it get right?** Per-word accuracy in
  `results/analysis.json`. Expect کیا / کب / کہاں / کتنا to do well — the answer
  span's *type* (number, date, place) is a strong surface cue. Expect کیوں and
  کیسے to fail, because the cause or manner is not recoverable from the marked
  span alone.
- **Why does Wiki-UQA drop?** UQA is machine-translated SQuAD, so the model
  learns translationese and SQuAD's question-template distribution. Wiki-UQA is
  natively Urdu, so it is a genuine distribution shift, not just a harder
  domain. That is the honest limitation of training on translated data.

## Qualitative samples (Section 4.6)

Pick ten from `results/samples.tsv` — five good, five bad. For each bad one name
the failure type: wrong question word, hallucinated entity, `<unk>`, repetition,
copied the sentence. `src/analyze.py` reports copy rate, repetition rate and the
five cases where beam hurt most, which is where the bad examples usually live.
