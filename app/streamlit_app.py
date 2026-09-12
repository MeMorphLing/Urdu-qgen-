"""Task 5 - front end.

    streamlit run app/streamlit_app.py

Paste an Urdu sentence, type the answer span exactly as it appears, and the app
wraps it in <ans>...</ans>, runs the trained model, and shows the greedy and
beam questions side by side.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sentencepiece as spm
import streamlit as st

from src.common import ANS_CLOSE, ANS_OPEN, get_device, load_cfg, load_checkpoint
from src.decode import beam_search_one, greedy_decode_one

st.set_page_config(page_title="Urdu question generator", page_icon="؟",
                   layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600&display=swap');
.urdu {
  font-family: 'Noto Nastaliq Urdu', serif;
  direction: rtl; text-align: right;
  font-size: 1.45rem; line-height: 2.6;
}
.question {
  border-right: 3px solid #1f5f5b;
  padding: 0.6rem 1.1rem 0.6rem 0.6rem;
  background: #f7f9f8;
}
.marked { background: #ffe9b8; padding: 0 .25rem; border-radius: 2px; }
.label { color: #5a6a68; font-size: .85rem; margin-bottom: .25rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def load(cfg_path="configs/base.yaml"):
    cfg = load_cfg(cfg_path)
    device = get_device()
    model, ckpt_cfg, ckpt = load_checkpoint(cfg["train"]["ckpt_path"], device)
    sp = spm.SentencePieceProcessor(model_file=cfg["tokenizer"]["model_file"])
    return model, sp, cfg, device, ckpt


def rtl(text, cls="urdu"):
    return f'<div class="{cls}">{text}</div>'


st.title("Urdu question generator")
st.write("Mark the answer inside a sentence and the model writes the question "
         "it answers.")

try:
    model, sp, cfg, device, ckpt = load()
except FileNotFoundError:
    st.error("No checkpoint found. Train the model first: `python -m src.train`.")
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

EXAMPLE_SENT = "\u062F\u0631\u06CC\u0627\u0626\u06D2 \u0633\u0646\u062F\u06BE \u0644\u06AF\u0628\u06BE\u06AF 3180 \u06A9\u0644\u0648\u0645\u06CC\u0679\u0631 \u0644\u0645\u0628\u0627 \u06C1\u06D2\u06D4"
EXAMPLE_ANS = "3180 \u06A9\u0644\u0648\u0645\u06CC\u0679\u0631"

sentence = st.text_area("Urdu sentence", EXAMPLE_SENT, height=110)
answer = st.text_input("Answer span (must appear in the sentence exactly)",
                       EXAMPLE_ANS)

if st.button("Generate question", type="primary"):
    sentence = " ".join(sentence.split())
    answer = " ".join(answer.split())

    if not sentence or not answer:
        st.warning("Enter a sentence and an answer span.")
        st.stop()
    idx = sentence.find(answer)
    if idx == -1:
        st.error("That answer span is not in the sentence. Copy it character for "
                 "character, including any diacritics.")
        st.stop()

    src = (sentence[:idx] + " " + ANS_OPEN + " " + answer + " " + ANS_CLOSE +
           " " + sentence[idx + len(answer):])
    src = " ".join(src.split())

    highlighted = (sentence[:idx] + f'<span class="marked">{answer}</span>' +
                   sentence[idx + len(answer):])
    st.markdown('<div class="label">Model input</div>', unsafe_allow_html=True)
    st.markdown(rtl(highlighted), unsafe_allow_html=True)

    src_ids = sp.encode(src)[:cfg["data"]["max_src_pieces"]]
    g_ids, attn = greedy_decode_one(model, src_ids, device, cfg["decode"]["max_len"])
    b_ids = beam_search_one(model, src_ids, device, beam_size,
                            cfg["decode"]["max_len"], alpha)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="label">Greedy</div>', unsafe_allow_html=True)
        st.markdown(rtl(sp.decode(g_ids), "urdu question"), unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="label">Beam (k={beam_size})</div>',
                    unsafe_allow_html=True)
        st.markdown(rtl(sp.decode(b_ids), "urdu question"), unsafe_allow_html=True)

    with st.expander("Subword pieces"):
        st.code(" ".join(sp.encode(src, out_type=str)))
