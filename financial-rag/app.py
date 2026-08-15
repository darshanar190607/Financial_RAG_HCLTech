import html
from datetime import datetime

import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from retriver import retrieve_chunks


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="HCLTech Financial RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# DESIGN SYSTEM — "FINANCIAL LEDGER" THEME
# Every color / font used below traces back to
# this token block. Change identity here only.
# ==========================================

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        :root {
            --canvas:      #F5F6F8;
            --canvas-alt:  #ECEEF2;
            --ink:         #0F1F3D;
            --ink-soft:    #4C5A75;
            --panel:       #0B1730;
            --panel-alt:   #142647;
            --panel-text:  #DCE3F0;
            --gold:        #C9962C;
            --gold-soft:   #E8C77A;
            --teal:        #1B7A72;
            --line:        #D8DEE9;
            --card:        #FFFFFF;
            --font-display:'Space Grotesk', sans-serif;
            --font-body:   'IBM Plex Sans', sans-serif;
            --font-mono:   'IBM Plex Mono', monospace;
        }

        /* ---------- base canvas ---------- */
        [data-testid="stAppViewContainer"] {
            background: var(--canvas);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        html, body, [class*="css"] {
            font-family: var(--font-body);
            color: var(--ink);
        }
        .block-container {
            padding-top: 2rem;
            max-width: 900px;
        }

        /* ---------- sidebar panel ---------- */
        [data-testid="stSidebar"] {
            background: var(--panel);
            border-right: 1px solid var(--panel-alt);
        }
        [data-testid="stSidebar"] * {
            color: var(--panel-text) !important;
            font-family: var(--font-body);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            font-family: var(--font-display) !important;
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: var(--panel-alt);
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: var(--panel-alt) !important;
            border-color: #223258 !important;
        }

        /* brand mark */
        .brand-row { display:flex; align-items:center; gap:10px; margin-bottom:2px; }
        .brand-mark {
            width:34px; height:34px; border-radius:8px;
            background: linear-gradient(135deg, var(--gold), var(--gold-soft));
            display:flex; align-items:center; justify-content:center;
            font-family: var(--font-display); font-weight:700; color: var(--panel);
            font-size: 15px; flex-shrink:0;
        }
        .brand-name { font-family: var(--font-display); font-size:17px; font-weight:700; color:#fff; line-height:1.1; }
        .brand-sub  { font-family: var(--font-mono); font-size:10.5px; letter-spacing:.12em; color: var(--gold-soft); text-transform:uppercase; }

        /* sidebar stat rows */
        .stat-panel { margin-top: 14px; }
        .stat-row {
            display:flex; justify-content:space-between; align-items:center;
            padding:8px 0; border-bottom:1px solid var(--panel-alt);
            font-family: var(--font-mono); font-size:12px;
        }
        .stat-row .label { color:#8FA0C2 !important; letter-spacing:.03em; }
        .stat-row .value { color:#fff !important; font-weight:500; }
        .status-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:#3FBF7F; margin-right:6px; box-shadow:0 0 6px #3FBF7F; }

        .sidebar-caption {
            font-family: var(--font-mono) !important; font-size: 10.5px !important;
            letter-spacing:.1em; text-transform:uppercase; color:#8FA0C2 !important;
            margin: 18px 0 6px 0 !important;
        }

        /* ---------- masthead ---------- */
        .masthead-rule { height:3px; background: linear-gradient(90deg, var(--gold), transparent); width:64px; margin-bottom:14px; }
        .eyebrow {
            font-family: var(--font-mono); font-size:12px; letter-spacing:.16em;
            text-transform:uppercase; color: var(--teal); margin-bottom:6px;
        }
        .headline {
            font-family: var(--font-display); font-weight:700; font-size:2.3rem;
            color: var(--ink); line-height:1.15; margin-bottom:6px;
        }
        .subhead {
            font-family: var(--font-body); font-size:15px; color: var(--ink-soft); margin-bottom:28px;
        }

        /* ---------- ledger-style input ---------- */
        /* Force background + text color explicitly (not "transparent") so the
           field stays legible even if the visitor's OS is in dark mode and
           Streamlit's own dark theme tries to darken the wrapper behind it. */
        [data-testid="stTextInput"] > div,
        [data-baseweb="input"],
        [data-baseweb="base-input"] {
            background: var(--canvas) !important;
            border-radius: 0 !important;
        }
        [data-testid="stTextInput"] input {
            font-family: var(--font-body) !important;
            font-size: 16px !important;
            padding: 12px 4px !important;
            border: none !important;
            border-bottom: 2px solid var(--line) !important;
            border-radius: 0 !important;
            background: var(--canvas) !important;
            color: var(--ink) !important;
            caret-color: var(--gold);
            -webkit-text-fill-color: var(--ink) !important;
        }
        [data-testid="stTextInput"] input:focus {
            border-bottom: 2px solid var(--gold) !important;
            box-shadow: none !important;
            background: var(--canvas) !important;
        }
        [data-testid="stTextInput"] input::placeholder {
            color: var(--ink-soft) !important;
            opacity: 0.55 !important;
        }
        [data-testid="stTextInput"] label {
            font-family: var(--font-mono) !important;
            font-size: 11.5px !important;
            letter-spacing:.08em; text-transform:uppercase;
            color: var(--ink-soft) !important;
        }

        /* ---------- buttons ---------- */
        button[kind="primary"], [data-testid="baseButton-primary"] {
            background: var(--gold) !important;
            color: var(--panel) !important;
            border: none !important;
            border-radius: 4px !important;
            font-family: var(--font-mono) !important;
            font-weight: 600 !important;
            letter-spacing: .06em;
            text-transform: uppercase;
            font-size: 13px !important;
            padding: 0.55rem 1.4rem !important;
        }
        button[kind="primary"]:hover, [data-testid="baseButton-primary"]:hover {
            background: var(--gold-soft) !important;
        }
        button[kind="secondary"], [data-testid="baseButton-secondary"] {
            background: transparent !important;
            border: 1px solid var(--line) !important;
            color: var(--ink-soft) !important;
            border-radius: 999px !important;
            font-family: var(--font-mono) !important;
            font-size: 12px !important;
            padding: 0.3rem 0.9rem !important;
        }
        button[kind="secondary"]:hover, [data-testid="baseButton-secondary"]:hover {
            border-color: var(--gold) !important;
            color: var(--gold) !important;
        }
        [data-testid="stSidebar"] button[kind="secondary"] {
            border: 1px solid #223258 !important;
            color: #DCE3F0 !important;
        }

        /* ---------- answer card ---------- */
        .answer-card {
            background: var(--card);
            border: 1px solid var(--line);
            border-left: 4px solid var(--gold);
            border-radius: 6px;
            padding: 22px 26px;
            margin: 18px 0 10px 0;
            box-shadow: 0 1px 3px rgba(15,31,61,0.06);
        }
        .answer-card.compact { padding: 16px 20px; margin: 6px 0; }
        .answer-meta {
            display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;
        }
        .answer-eyebrow {
            font-family: var(--font-mono); font-size:11px; letter-spacing:.14em;
            text-transform:uppercase; color: var(--teal);
        }
        .answer-quarter {
            font-family: var(--font-mono); font-size:11px; color: var(--ink-soft);
            border: 1px solid var(--line); border-radius:999px; padding:2px 10px;
        }
        .answer-question {
            font-family: var(--font-body); font-weight:500; font-size:14px;
            color: var(--ink-soft); margin-bottom: 10px;
        }
        .answer-text {
            font-family: var(--font-display); font-weight:500; font-size:19px;
            color: var(--ink); line-height:1.45;
        }

        /* ---------- source chips ---------- */
        .source-row { margin-top: 14px; display:flex; flex-wrap:wrap; gap:8px; }
        .src-chip {
            font-family: var(--font-mono); font-size:11.5px; color: var(--ink-soft);
            border: 1px solid var(--line); border-radius: 999px; padding: 4px 12px;
            background: var(--canvas-alt);
        }
        .src-chip .src-page { color: var(--teal); font-weight:600; margin-left:4px; }

        /* ---------- session ledger ---------- */
        .ledger-heading {
            font-family: var(--font-display); font-weight:600; font-size:15px;
            color: var(--ink); margin: 30px 0 6px 0;
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--line) !important;
            border-radius: 6px !important;
            background: var(--card);
        }
        [data-testid="stExpander"] summary {
            font-family: var(--font-mono) !important;
            font-size: 12.5px !important;
            color: var(--ink) !important;
        }

        /* ---------- footer ---------- */
        .app-footer {
            margin-top: 40px; padding-top: 14px; border-top: 1px solid var(--line);
            font-family: var(--font-mono); font-size: 11px; color: var(--ink-soft);
            text-align:center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# LOAD LLM  (unchanged from original)
# ==========================================

@st.cache_resource
def load_model():

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    return tokenizer, model


# ==========================================
# QUARTER -> SOURCE FILE MAP  (unchanged)
# ==========================================

QUARTER_TO_FILE = {
    "Q1 FY26": "HCLTech_Q1_FY26.pdf",
    "Q2 FY26": "HCLTech_Q2_FY26.pdf",
    "Q3 FY26": "HCLTech_Q3_FY26.pdf",
    "Q4 FY26": "HCLTech_Q4_FY26.pdf"
}

EXAMPLE_QUESTIONS = [
    "What was revenue from operations in Q1 FY26?",
    "How did net profit change year-on-year?",
    "What was the operating margin in Q3 FY26?",
    "Summarize headcount trends across all quarters",
]


# ==========================================
# CORE RAG CALL — retrieval + generation logic
# is IDENTICAL to the original script, just
# wrapped in a function so the UI layer stays
# clean and reusable for history entries.
# ==========================================

def answer_question(tokenizer, model, question: str, quarter: str) -> dict:

    source = None if quarter == "All Quarters" else QUARTER_TO_FILE[quarter]

    results = retrieve_chunks(
        question=question,
        source=source,
        top_k=2
    )

    # ---- build context (unchanged) ----
    context_parts = []
    for i in range(len(results["documents"][0])):
        text = results["documents"][0][i]
        source_name = results["metadatas"][0][i]["source"]
        page = results["metadatas"][0][i]["page"]
        context_parts.append(
            f"Source: {source_name}\nPage: {page}\nContent:\n{text}"
        )
    context = "\n\n".join(context_parts)

    # ---- build prompt (unchanged) ----
    prompt = f"""
Context:
{context}

Question:
{question}

Answer the question using only the context.
Give a short factual answer.
Do not use information outside the context.

Answer:
"""

    # ---- generate (unchanged) ----
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )
    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # ---- dedupe sources, preserving order ----
    seen = set()
    sources = []
    for i in range(len(results["documents"][0])):
        source_name = results["metadatas"][0][i]["source"]
        page = results["metadatas"][0][i]["page"]
        key = (source_name, page)
        if key not in seen:
            sources.append(key)
            seen.add(key)

    return {
        "question": question,
        "answer": answer,
        "quarter": quarter,
        "sources": sources,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }


# ==========================================
# RENDER HELPERS
# ==========================================

def esc(text: str) -> str:
    """Escape user/model text before injecting into HTML."""
    return html.escape(str(text))


def render_answer_card(entry: dict, entry_no: int, featured: bool = False):

    card_class = "answer-card" if featured else "answer-card compact"
    eyebrow = "LATEST ANSWER" if featured else f"LEDGER ENTRY {entry_no:03d}"

    chips = "".join(
        f'<span class="src-chip">📄 {esc(name)} <span class="src-page">P.{esc(page)}</span></span>'
        for name, page in entry["sources"]
    ) or '<span class="src-chip">No sources retrieved</span>'

    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="answer-meta">
                <span class="answer-eyebrow">{eyebrow} · {esc(entry['timestamp'])}</span>
                <span class="answer-quarter">{esc(entry['quarter'])}</span>
            </div>
            <div class="answer-question">{esc(entry['question'])}</div>
            <div class="answer-text">{esc(entry['answer'])}</div>
            <div class="source-row">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar():

    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-mark">HT</div>
            <div>
                <div class="brand-name">HCLTech</div>
                <div class="brand-sub">Financial Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-caption">Filter</div>', unsafe_allow_html=True)
    quarter = st.selectbox(
        "Reporting quarter",
        ["Q1 FY26", "Q2 FY26", "Q3 FY26", "Q4 FY26", "All Quarters"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="sidebar-caption">Quick questions</div>', unsafe_allow_html=True)
    for q in EXAMPLE_QUESTIONS:
        st.button(q, key=f"eq_{q}", on_click=set_pending_question, args=(q,), type="secondary")

    st.markdown('<div class="sidebar-caption">System status</div>', unsafe_allow_html=True)
    n_queries = len(st.session_state.history)
    st.markdown(
        f"""
        <div class="stat-panel">
            <div class="stat-row"><span class="label"><span class="status-dot"></span>Status</span><span class="value">Ready</span></div>
            <div class="stat-row"><span class="label">Model</span><span class="value">Flan-T5-Base</span></div>
            <div class="stat-row"><span class="label">Retrieval depth</span><span class="value">Top-2 passages</span></div>
            <div class="stat-row"><span class="label">Coverage</span><span class="value">4 quarters · FY26</span></div>
            <div class="stat-row"><span class="label">Session queries</span><span class="value">{n_queries}</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    if st.button("Clear session ledger", type="secondary"):
        st.session_state.history = []
        st.rerun()

    st.markdown(
        """
        <div style="margin-top:24px; font-family: var(--font-mono); font-size:10.5px; color:#8FA0C2; line-height:1.5;">
        Answers are generated from indexed HCLTech quarterly filings only.
        Verify figures against the original filing before external use.
        </div>
        """,
        unsafe_allow_html=True
    )

    return quarter


def set_pending_question(q: str):
    st.session_state.question_input = q


# ==========================================
# SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""


# ==========================================
# APP BODY
# ==========================================

inject_css()
tokenizer, model = load_model()

with st.sidebar:
    selected_quarter = render_sidebar()

st.markdown('<div class="masthead-rule"></div>', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">Quarterly Earnings Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="headline">Ask HCLTech\'s financial results</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subhead">Answers are retrieved from indexed quarterly filings and generated with citations to source page.</div>',
    unsafe_allow_html=True
)

question = st.text_input(
    "Question",
    key="question_input",
    placeholder="e.g. What was HCLTech's revenue from operations in Q1 FY26?"
)

ask_col, _ = st.columns([1, 5])
with ask_col:
    ask_clicked = st.button("Ask", type="primary")

if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching financial documents and generating answer..."):
            entry = answer_question(tokenizer, model, question, selected_quarter)
        st.session_state.history.append(entry)

# ---- render latest answer, featured ----
if st.session_state.history:
    latest = st.session_state.history[-1]
    render_answer_card(latest, entry_no=len(st.session_state.history), featured=True)

# ---- render past entries as a numbered ledger ----
if len(st.session_state.history) > 1:
    st.markdown('<div class="ledger-heading">Session Ledger</div>', unsafe_allow_html=True)
    past_entries = list(enumerate(st.session_state.history[:-1], start=1))
    for entry_no, entry in reversed(past_entries):
        with st.expander(f"{entry_no:03d} — {entry['question']}"):
            render_answer_card(entry, entry_no=entry_no, featured=False)

st.markdown(
    '<div class="app-footer">HCLTech Financial RAG · Retrieval-Augmented Generation over indexed quarterly filings · For informational use only</div>',
    unsafe_allow_html=True
)