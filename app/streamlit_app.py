"""Task 5 - front end.

    streamlit run app/streamlit_app.py

Pick a preset or paste an Urdu sentence, give the answer span exactly as it
appears, and the app wraps it in <ans>...</ans>, runs the trained model, and
shows the greedy and beam questions side by side plus the attention map.

Presets are mined from data/valid.tsv at load time: the reference question's
interrogative (kitna / kab / kaun / kahan) labels the answer type far more
reliably than guessing it from the span itself.
"""
import difflib
import html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sentencepiece as spm
import streamlit as st

from src.common import ANS_CLOSE, ANS_OPEN, get_device, load_cfg, load_checkpoint
from src.dataset import read_tsv
from src.decode import beam_search_one, greedy_decode_one
from src.viz import shape, urdu_font

st.set_page_config(page_title="Urdu question generator", page_icon="؟",
                   layout="wide")

# The Google font covers the Urdu we render ourselves; the max-width clamp keeps
# the wide layout from stretching lines past a comfortable measure.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600&display=swap');
.stMainBlockContainer { max-width: 1180px; padding-top: 2.6rem; }
.urdu {
  font-family: 'Noto Nastaliq Urdu', serif;
  direction: rtl; text-align: right;
  font-size: 1.4rem; line-height: 2.7;
}
.st-key-sentence textarea, .st-key-answer input, .stTextArea textarea, .stTextInput input {
  font-family: 'Noto Nastaliq Urdu', serif !important;
  direction: rtl; text-align: right;
  font-size: 1.15rem !important; line-height: 2.7 !important;
  padding: 0.75rem 0.9rem !important;
}
.card {
  border: 1px solid #dde4e1; border-right: 4px solid #1f5f5b;
  border-radius: 8px; background: #ffffff;
  padding: 0.7rem 1.2rem 0.9rem 1.2rem;
}
.card.plain { border-right-color: #c3cfca; }
.marked {
  background: #ffe9b8; padding: 0 .3rem; border-radius: 3px;
  box-shadow: inset 0 -2px 0 #e8c67a;
}
.label {
  color: #5a6a68; font-size: .78rem; letter-spacing: .06em;
  text-transform: uppercase; margin-bottom: .1rem;
}
.hint { color: #7b8784; font-size: .85rem; }
</style>
""", unsafe_allow_html=True)

ANS_RE = re.compile(re.escape(ANS_OPEN) + r"(.*?)" + re.escape(ANS_CLOSE))

# Urdu interrogative -> answer type it implies. First hit wins.
PRESET_KINDS = [
    ("Number", ("کتنا", "کتنی",
                "کتنے")),
    ("Date", ("کب",)),
    ("Person", ("کون", "کس نے")),
    ("Place", ("کہاں",)),
]

FALLBACK_SENT = "دریائے سندھ لگبھگ 3180 کلومیٹر لمبا ہے۔"
FALLBACK_ANS = "3180 کلومیٹر"


@st.cache_resource(show_spinner="Loading model...")
def load(cfg_path="configs/base.yaml"):
    cfg = load_cfg(cfg_path)
    device = get_device()
    model, ckpt_cfg, ckpt = load_checkpoint(cfg["train"]["ckpt_path"], device)
    sp = spm.SentencePieceProcessor(model_file=cfg["tokenizer"]["model_file"])
    return model, sp, cfg, device, ckpt


@st.cache_data(show_spinner=False)
def load_presets(path="data/valid.tsv", min_words=8, max_words=24):
    """One example per answer type, taken from the validation split."""
    try:
        rows = read_tsv(path)
    except OSError:
        return {}

    found = {}
    for src, question in rows:
        m = ANS_RE.search(src)
        if not m:
            continue
        span = " ".join(m.group(1).split())
        sentence = " ".join(ANS_RE.sub(lambda mm: mm.group(1), src).split())
        if not span or not (min_words <= len(sentence.split()) <= max_words):
            continue
        if not 1 <= len(span.split()) <= 5:
            continue
        # UQA is sentence-split SQuAD, so plenty of rows are mid-sentence
        # fragments. A letter start and a non-initial span read as real
        # sentences in the screenshot; the rest do not.
        if not sentence[0].isalpha() or sentence.find(span) < 1:
            continue
        for kind, qwords in PRESET_KINDS:
            if kind not in found and any(w in question for w in qwords):
                found[kind] = (sentence, span)
        if len(found) == len(PRESET_KINDS):
            break
    return {kind: found[kind] for kind, _ in PRESET_KINDS if kind in found}


def rtl(text, cls="urdu"):
    return f'<div class="{cls}">{html.escape(text)}</div>'


def card(label, body_html, plain=False):
    cls = "card plain" if plain else "card"
    return (f'<div class="{cls}"><div class="label">{label}</div>'
            f'{body_html}</div>')


def near_miss(sentence, answer):
    """Closest span in the sentence when an exact match fails, else None."""
    toks = answer.split()
    if not toks:
        return None
    # Whitespace-insensitive first: the cheapest and most common near-miss.
    m = re.search(r"\s*".join(map(re.escape, toks)), sentence)
    if m:
        return m.group(0)
    # Then same-word-count windows, which catches ZWNJ, diacritics and typos.
    words = sentence.split()
    n = len(toks)
    windows = {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}
    close = difflib.get_close_matches(answer, sorted(windows), n=1, cutoff=0.75)
    return close[0] if close else None


def attention_figure(attn, src_pieces, out_pieces):
    """Heat-map of attention. Returns (figure, legend or None if a font loaded)."""
    fp = urdu_font()

    def clean(pieces):
        return [p.replace("▁", " ").strip() or "_" for p in pieces]

    fig, ax = plt.subplots(figsize=(min(14, max(6, len(src_pieces) * 0.42)),
                                    min(9, max(3, len(out_pieces) * 0.42))))
    im = ax.imshow(attn[:len(out_pieces)], aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(src_pieces)))
    ax.set_yticks(range(len(out_pieces)))

    legend = None
    if fp is not None:
        ax.set_xticklabels([shape(p) for p in clean(src_pieces)], rotation=90,
                           fontproperties=fp)
        ax.set_yticklabels([shape(p) for p in clean(out_pieces)], fontproperties=fp)
    else:
        # No Urdu .ttf anywhere: indices beat a grid of tofu boxes.
        ax.set_xticklabels(range(len(src_pieces)), rotation=90, fontsize=7)
        ax.set_yticklabels(range(len(out_pieces)), fontsize=7)
        legend = ("source: " + "  ".join(f"{i}:{p}" for i, p in enumerate(clean(src_pieces)))
                  + "\n\noutput: " + "  ".join(f"{i}:{p}" for i, p in enumerate(clean(out_pieces))))

    ax.set_xlabel("source pieces")
    ax.set_ylabel("generated pieces")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    return fig, legend


st.title("Urdu question generator")
st.caption("Mark the answer inside a sentence and the model writes the question "
           "it answers. RNN encoder-decoder with Bahdanau attention, trained "
           "from scratch on UQA.")

try:
    model, sp, cfg, device, ckpt = load()
except (FileNotFoundError, OSError, RuntimeError) as e:
    st.error("The trained model is not available yet, so generation is disabled.")
    st.markdown("Run the pipeline first:")
    st.code("python -m src.prepare_data --config configs/base.yaml\n"
            "python -m src.train_tokenizer --config configs/base.yaml\n"
            "python -m src.train --config configs/base.yaml", language="bash")
    st.caption(f"{type(e).__name__}: {e}")
    st.stop()

with st.sidebar:
    st.caption("Model")
    m = ckpt["cfg"]["model"]
    st.write(f"{m['rnn_type'].upper()} encoder-decoder + Bahdanau attention")
    st.write(f"emb {m['emb_dim']} / hidden {m['hid_dim']} / {m['enc_layers']} layers")
    st.write(f"best epoch {ckpt['epoch']}, valid loss {ckpt['val_loss']:.3f}")
    beam_size = st.slider("Beam size", 1, 10, cfg["decode"]["beam_size"])
    alpha = st.slider("Length penalty alpha", 0.0, 1.5,
                      float(cfg["decode"]["length_alpha"]), 0.1)

presets = load_presets()
st.session_state.setdefault("sentence", FALLBACK_SENT)
st.session_state.setdefault("answer", FALLBACK_ANS)


def apply_preset():
    choice = st.session_state.preset
    if choice in presets:
        st.session_state.sentence, st.session_state.answer = presets[choice]


def apply_suggestion(span):
    st.session_state.answer = span
    st.session_state.autorun = True


left, right = st.columns([3, 2], gap="large")
with left:
    st.selectbox("Example", ["Custom"] + list(presets), key="preset",
                 on_change=apply_preset,
                 help="Validation-set sentences with different answer types.")
    st.text_area("Urdu sentence", key="sentence", height=170)
with right:
    st.text_input("Answer span", key="answer")
    st.markdown('<div class="hint">Must appear in the sentence exactly, '
                'character for character.</div>', unsafe_allow_html=True)
    st.write("")
    go = st.button("Generate question", type="primary", use_container_width=True)

autorun = st.session_state.pop("autorun", False)

if go or autorun:
    sentence = " ".join(st.session_state.sentence.split())
    answer = " ".join(st.session_state.answer.split())

    if not sentence or not answer:
        st.warning("Enter a sentence and an answer span.")
        st.stop()

    idx = sentence.find(answer)
    if idx == -1:
        st.error("That answer span does not appear in the sentence.")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown(card("Searched for", rtl(answer), plain=True),
                        unsafe_allow_html=True)
        suggestion = near_miss(sentence, answer)
        if suggestion:
            with c2:
                st.markdown(card("Closest match in the sentence", rtl(suggestion)),
                            unsafe_allow_html=True)
                st.button("Use this span", on_click=apply_suggestion,
                          args=(suggestion,))
        else:
            st.caption("No close match either - copy the span straight out of "
                       "the sentence, including any diacritics.")
        st.stop()

    src = (sentence[:idx] + " " + ANS_OPEN + " " + answer + " " + ANS_CLOSE +
           " " + sentence[idx + len(answer):])
    src = " ".join(src.split())

    highlighted = (html.escape(sentence[:idx]) +
                   f'<span class="marked">{html.escape(answer)}</span>' +
                   html.escape(sentence[idx + len(answer):]))
    st.markdown(card("Model input", f'<div class="urdu">{highlighted}</div>',
                     plain=True), unsafe_allow_html=True)
    st.write("")

    src_ids = sp.encode(src)[:cfg["data"]["max_src_pieces"]]
    g_ids, attn = greedy_decode_one(model, src_ids, device, cfg["decode"]["max_len"])
    b_ids = beam_search_one(model, src_ids, device, beam_size,
                            cfg["decode"]["max_len"], alpha)
    greedy_q, beam_q = sp.decode(g_ids), sp.decode(b_ids)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(card("Greedy", rtl(greedy_q)), unsafe_allow_html=True)
    with c2:
        st.markdown(card(f"Beam &nbsp;k={beam_size} &nbsp;alpha={alpha:g}",
                         rtl(beam_q)), unsafe_allow_html=True)
    st.caption("Identical output" if greedy_q == beam_q
               else "Beam and greedy disagree on this sentence.")

    with st.expander("Subword pieces"):
        st.code(" ".join(sp.encode(src, out_type=str)))

    with st.expander("Attention heat-map"):
        if attn is None or not g_ids:
            st.caption("No attention to show - the model emitted nothing.")
        else:
            fig, legend = attention_figure(attn, sp.id_to_piece(src_ids),
                                           sp.id_to_piece(g_ids))
            st.pyplot(fig)
            plt.close(fig)
            if legend:
                st.caption("No Urdu font in fonts/ - axes are indexed. Drop a "
                           ".ttf there for Nastaliq labels.")
                st.code(legend)
