"""Streamlit front end:  streamlit run app.py"""
import base64
from html import escape

import pymupdf
import streamlit as st

from common import ROOT
from pipeline import Assistant
from retrieval import FILTERS

FOUND_BY = {"both": "keyword and meaning search", "keyword": "keyword search", "semantic": "meaning search"}
# A blood drop: the brand mark.
MARK = ('<svg class="mark" viewBox="0 0 44 44" aria-hidden="true"><path d="M22 4 C22 4 8 20 8 29 a14 14 0 0 0 28 0 '
        'C36 20 22 4 22 4 Z" fill="#2B4C9B"/><path d="M15 29 a7 7 0 0 0 7 7" stroke="#fff" stroke-width="2.5" '
        'fill="none" stroke-linecap="round"/></svg>')

st.set_page_config(page_title="Diabetes Guideline Assistant", page_icon="🩺", layout="centered")
st.markdown(f"<style>{(ROOT / 'style.css').read_text()}</style>", unsafe_allow_html=True)


def doctor_image():
    """Use assets/doctor.jpg/.png/.webp if you add one, otherwise the bundled illustration."""
    for name, mime in [("doctor.jpg", "jpeg"), ("doctor.jpeg", "jpeg"), ("doctor.png", "png"),
                       ("doctor.webp", "webp"), ("doctor.svg", "svg+xml")]:
        path = ROOT / "assets" / name
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            return f'<img class="doctor" src="data:image/{mime};base64,{data}" alt="Doctor">'
    return ""


def h(text):
    """Escape model/guideline text for HTML; keep line breaks without breaking the markdown block."""
    return escape(str(text)).replace("\n", "<br>")


def chip(guideline):
    return f'<span class="chip {"icmr" if guideline.startswith("ICMR") else "ada"}">{escape(guideline)}</span>'


@st.cache_resource(show_spinner="Loading the guideline library...")
def load_assistant():
    return Assistant()


@st.cache_data(max_entries=64)
def render_page(file, page, quote):
    """PNG of the cited PDF page, with the quote highlighted when PyMuPDF can locate it."""
    with pymupdf.open(ROOT / file) as doc:
        pdf_page = doc[page - 1]
        words = quote.split()
        for needle in (quote, " ".join(words[:8])):  # fall back to the first words of the quote
            rects = pdf_page.search_for(needle) if needle else []
            if rects:
                pdf_page.add_highlight_annot(rects)
                break
        return pdf_page.get_pixmap(dpi=110).tobytes("png")


# ---------- Header ----------
st.markdown(f"""
<div class="letterhead">{MARK}<h1>Diabetes Guideline Assistant</h1></div>
<div class="notice">Answers are drafted by AI. Always verify with the source.</div>
""", unsafe_allow_html=True)

try:
    assistant = load_assistant()
except RuntimeError as e:  # e.g. .env missing
    st.error(str(e))
    st.stop()

# ---------- Question ----------
with st.container(key="consult"):
    left, right = st.columns([1, 3.2], gap="large")
    with left:
        st.markdown(doctor_image(), unsafe_allow_html=True)
    with right:
        st.markdown('<div class="ask-title">Ask a question</div>', unsafe_allow_html=True)
        with st.form("ask", border=False):
            question = st.text_area("Question", height=110, label_visibility="collapsed",
                                    placeholder="Ask a question or describe the patient (no names)")
            col_filter, col_button = st.columns([2, 1], vertical_alignment="bottom")
            filter_name = col_filter.selectbox("Search in", FILTERS, label_visibility="collapsed")
            submitted = col_button.form_submit_button("Ask", type="primary", use_container_width=True)

if submitted:
    if not question.strip():
        st.warning("Type a question or patient details first.")
    else:
        with st.spinner("Reading the guidelines and checking every quote..."):
            try:
                st.session_state.result = assistant.ask(question.strip(), filter_name)
            except Exception as e:  # API errors, rate limits, invalid JSON from the model, ...
                st.session_state.pop("result", None)
                msg = str(e)
                if "PerDay" in msg:
                    msg = ("Today's free AI limit is used up (Gemini free tier has a daily limit per model). "
                           "It resets at midnight Pacific time, or set a different MODEL_NAME in .env.")
                elif "429" in msg or "quota" in msg.lower():
                    msg = "The AI service limit was reached (free tier: about 5 questions per minute). Wait a minute and ask again."
                st.error(f"No answer this time. {msg}")

result = st.session_state.get("result")

# ---------- Answer ----------
if result:
    answer = result["answer"]
    st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)
    if answer["not_found"] and not answer["recommendations"]:
        st.markdown('<div class="summary empty">Not found in the provided guidelines.'
                    '<span class="hint">Try rephrasing the question.</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="summary">{h(answer["answer_summary"])}</div>', unsafe_allow_html=True)

    for rec in answer["recommendations"]:
        v = rec["verification"]
        page = rec.get("page", "?")
        if v["verified"]:
            status = f'<span class="status ok" title="{v["score"]}% match on the cited page">✔ Verified</span>'
        else:
            status = f'<span class="status flag" title="{h(v["reason"])}">⚠ Check quote</span>'
        st.markdown(f"""
<div class="rec">
  <div class="rec-text">{h(rec.get("text", ""))}</div>
  <blockquote>“{h(rec.get("quote", ""))}”</blockquote>
  <div class="rec-foot">{chip(str(rec.get("guideline", "?")))}
    <span>{escape(str(rec.get("document", "")))} · p. {escape(str(page))}</span>{status}</div>
</div>""", unsafe_allow_html=True)
        if v.get("file"):
            with st.expander(f"View page {v['page']}"):
                try:
                    st.image(render_page(v["file"], v["page"], rec.get("quote", "")))
                except (IndexError, ValueError):
                    st.write("This page could not be displayed.")

    if answer["differences_between_guidelines"]:
        items = "".join(f"<li>{h(d)}</li>" for d in answer["differences_between_guidelines"])
        st.markdown(f'<div class="differ"><h3>ADA and ICMR differ</h3><ul>{items}</ul></div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="section-title">Sources</div>', unsafe_allow_html=True)
    cited = {}
    for c in result["retrieved"]:
        cited.setdefault((c["guideline"], c["document"]), []).append(c["page"])
    st.markdown('<div class="source-list">' + "".join(
        f'<div>{chip(g)}<span>{escape(d)} · p. {", ".join(str(p) for p in sorted(set(pages)))}</span></div>'
        for (g, d), pages in cited.items()) + "</div>", unsafe_allow_html=True)
    with st.expander("Passages and search details"):
        searched = "".join(f"<li>{h(q)}</li>" for q in result.get("queries", [result["question"]]))
        rows = [f'<div class="searched"><b>Searched for</b><ul>{searched}</ul></div>']
        for c in result["retrieved"]:
            ranks = f"keyword rank {c['keyword_rank'] or '–'}, meaning rank {c['semantic_rank'] or '–'}"
            section = f"Section: {escape(c['section'])}. " if c["section"] else ""
            rows.append(f"""
<div class="source">
  <div class="source-head">{chip(c['guideline'])}<span class="where">{escape(c['document'])}, page {c['page']}</span></div>
  <div class="source-meta">{section}Found by {FOUND_BY[c['found_by']]} for {len(c.get('queries', [1]))} of the searches (fusion score {c['rrf_score']}; best {ranks}).</div>
  <div class="source-text">{h(c['text'])}</div>
</div>""")
        st.markdown("".join(rows), unsafe_allow_html=True)

    t, tok = result["timing"], result["tokens"]
    st.markdown(f'<div class="footer">{t["total_s"]} s · {tok.get("total", "?")} tokens · {escape(result["model"])}</div>',
                unsafe_allow_html=True)
