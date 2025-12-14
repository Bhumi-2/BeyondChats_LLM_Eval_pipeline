#!/usr/bin/env python3
import time
import math
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

import streamlit as st


# ============================================================
# Metrics (same logic as evaluator.py)
# ============================================================

_WORD = re.compile(r"[A-Za-z0-9']+")

NUM_RE = re.compile(
    r"(?:₹|rs\.?)?\s*(?:\d{1,3}(?:,\d{3})+|\d{3,})(?:\.\d+)?",
    re.IGNORECASE
)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(text or "")]


def bow_vector(tokens: List[str]) -> Dict[str, float]:
    v: Dict[str, float] = {}
    for t in tokens:
        v[t] = v.get(t, 0.0) + 1.0
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    for k in list(v.keys()):
        v[k] /= norm
    return v


def cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(a.get(k, 0.0) * b.get(k, 0.0) for k in a.keys())


def relevance_score(response: str, contexts: List[str]) -> float:
    r = bow_vector(tokenize(response))
    sims = [cosine_sparse(r, bow_vector(tokenize(c))) for c in contexts]
    return float(max(sims) if sims else 0.0)


def completeness_score(response: str, user_query: str, contexts: List[str]) -> float:
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are", "was", "were", "be",
        "i", "you", "we", "they", "it", "this", "that", "these", "those", "as", "at", "by", "with",
        "from", "can", "could", "should", "would", "will", "may", "might", "do", "does", "did",
        "please", "tell", "me", "my"
    }
    q_tokens = [t for t in tokenize(user_query) if len(t) >= 3 and t not in stop]
    if not q_tokens:
        return 0.0
    r_set = set(tokenize(response))
    covered = sum(1 for t in q_tokens if t in r_set)
    rel = relevance_score(response, contexts)
    base = covered / max(len(set(q_tokens)), 1)
    return float(min(1.0, base * 0.9 + rel * 0.1))


def hallucination_score(response: str, contexts: List[str], threshold: float = 0.30) -> Tuple[float, List[Dict[str, Any]]]:
    ctx_tokens = set()
    for c in contexts:
        ctx_tokens.update(tokenize(c))

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response or "") if s.strip()]
    if not sentences:
        return 0.0, []

    flagged = []
    supported = 0
    for s in sentences:
        toks = [t for t in tokenize(s) if len(t) >= 3]
        if not toks:
            supported += 1
            continue
        overlap = sum(1 for t in set(toks) if t in ctx_tokens)
        ratio = overlap / max(len(set(toks)), 1)
        if ratio >= threshold:
            supported += 1
        else:
            flagged.append({"sentence": s, "overlap_ratio": round(ratio, 3), "overlap_terms": overlap})

    return float(supported / len(sentences)), flagged


def _norm_num(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"(₹|rs\.?)", "", s)
    s = re.sub(r"\s+", "", s)
    s = s.replace(",", "")
    return s


def unsupported_numeric_claims(response: str, contexts: List[str]) -> List[str]:
    response = re.sub(r"\b\d{1,2}:\d{2}\b", "", response or "")

    ctx_nums = set()
    for c in contexts:
        for n in NUM_RE.findall(c or ""):
            nn = _norm_num(n)
            if nn:
                ctx_nums.add(nn)

    found = []
    for n in NUM_RE.findall(response or ""):
        nn = _norm_num(n)
        if nn and nn not in ctx_nums:
            found.append(n.strip())

    seen, out = set(), []
    for x in found:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def numeric_penalty(unsupported_nums: List[str]) -> float:
    if not unsupported_nums:
        return 0.0
    return 0.5


@dataclass
class CostModel:
    usd_per_1k_input_tokens: float = 0.0
    usd_per_1k_output_tokens: float = 0.0

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens / 1000.0) * self.usd_per_1k_input_tokens
            + (output_tokens / 1000.0) * self.usd_per_1k_output_tokens
        )


def parse_contexts(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x) for x in data if str(x).strip()]
        except Exception:
            pass
    return [line.strip() for line in raw.splitlines() if line.strip()]


def build_report(user_query: str, ai_answer: str, contexts: List[str], threshold: float, cm: CostModel) -> Dict[str, Any]:
    t0 = time.perf_counter()

    rel = relevance_score(ai_answer, contexts)
    comp = completeness_score(ai_answer, user_query, contexts) if user_query else 0.0

    lexical_support, flagged = hallucination_score(ai_answer, contexts, threshold=threshold)
    nums_unsupported = unsupported_numeric_claims(ai_answer, contexts)

    pen = numeric_penalty(nums_unsupported)
    factual_support = max(0.0, min(1.0, lexical_support - pen))

    in_tokens = estimate_tokens((user_query or "") + " " + " ".join(contexts[:3]))
    out_tokens = estimate_tokens(ai_answer or "")
    est_cost = cm.estimate(in_tokens, out_tokens)

    latency = time.perf_counter() - t0

    return {
        "user_query": user_query,
        "assistant_response": ai_answer,
        "contexts_count": len(contexts),
        "relevance_score": round(rel, 3),
        "completeness_score": round(comp, 3),

        "lexical_support_score": round(lexical_support, 3),
        "numeric_penalty": round(pen, 3),
        "factual_support_score": round(factual_support, 3),

        "potentially_unsupported_sentences": flagged[:10],
        "unsupported_numeric_claims": nums_unsupported,
        "estimated_tokens": {"input": in_tokens, "output": out_tokens},
        "estimated_cost_usd": round(est_cost, 6),
        "latency_seconds": round(latency, 4),
    }


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="LLM Evaluation Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; max-width: 1250px; }
      h1, h2, h3 { letter-spacing: -0.02em; }
      .muted { color: rgba(255,255,255,0.65); }
      .card {
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 16px 16px;
      }
      .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
        font-size: 12px;
        color: rgba(255,255,255,0.78);
        margin-right: 8px;
        margin-bottom: 8px;
      }
      .smallcap { font-size: 12px; color: rgba(255,255,255,0.65); }
      .stTextArea textarea { border-radius: 14px; }
      .stButton button { border-radius: 12px; padding: 0.6rem 1rem; font-weight: 600; }
      .stDownloadButton button { border-radius: 12px; padding: 0.6rem 1rem; font-weight: 600; }
      div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 14px;
        border-radius: 16px;
      }
      .hr { height: 1px; background: rgba(255,255,255,0.08); margin: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## LLM Evaluation Dashboard (Real-time Heuristics)")
st.markdown(
    '<div class="muted">Paste a user query, retrieved context(s), and an AI answer. '
    "The app computes relevance, completeness, lexical support, numeric hallucinations, latency, and estimated cost.</div>",
    unsafe_allow_html=True
)

# Session state init (prevents autofill clearing on rerun)
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "ai_answer" not in st.session_state:
    st.session_state.ai_answer = ""
if "contexts_raw" not in st.session_state:
    st.session_state.contexts_raw = ""
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("### Settings")
    st.markdown('<div class="smallcap">Tune thresholds and cost model.</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("#### Cost model")
    in_cost = st.number_input("USD per 1K input tokens", min_value=0.0, value=0.003, step=0.001, format="%.4f")
    out_cost = st.number_input("USD per 1K output tokens", min_value=0.0, value=0.006, step=0.001, format="%.4f")

    st.divider()
    st.markdown("#### Thresholds")
    sent_support_threshold = st.slider(
        "Sentence support threshold (overlap ratio)",
        0.05, 0.60, 0.30, 0.01
    )

    st.divider()
    st.markdown("#### Quick test cases")
    st.markdown('<div class="smallcap">Autofill inputs to demo normal vs hallucination.</div>', unsafe_allow_html=True)

    demo_good = st.button("Fill: Good answer")
    demo_hallu = st.button("Fill: Hallucinated number")
    demo_incomplete = st.button("Fill: Low completeness")

    if demo_good:
        st.session_state.user_query = "How much does an online consultation with Dr. Malpani cost and when is it available?"
        st.session_state.contexts_raw = "\n".join([
            "Dr. Malpani offers online consultations via Zoom.",
            "Each consultation lasts 30 minutes.",
            "The consultation fee is ₹3,000.",
            "Consultations are available Monday to Saturday between 2 PM and 4 PM."
        ])
        st.session_state.ai_answer = (
            "Dr. Malpani provides 30-minute online consultations via Zoom for ₹3,000. "
            "These sessions are available Monday to Saturday between 2:00 PM and 4:00 PM."
        )

    if demo_hallu:
        st.session_state.user_query = "How much does an online consultation cost and when is it available?"
        st.session_state.contexts_raw = "\n".join([
            "Online consultations are available Monday to Saturday between 2 PM and 4 PM.",
            "The fee for an online consultation is ₹3,000."
        ])
        st.session_state.ai_answer = "The consultation costs ₹10,000 and is available 24/7."

    if demo_incomplete:
        st.session_state.user_query = "How much does an online consultation cost and when is it available?"
        st.session_state.contexts_raw = "\n".join([
            "The fee for an online consultation is ₹3,000.",
            "Consultations are available Monday to Saturday between 2 PM and 4 PM."
        ])
        st.session_state.ai_answer = "The online consultation costs ₹3,000."


left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Inputs")
    st.text_area(
        "User query",
        height=110,
        key="user_query",
        placeholder="Enter the user question..."
    )
    st.text_area(
        "AI answer",
        height=190,
        key="ai_answer",
        placeholder="Paste the model/assistant response..."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Retrieved context(s)")
    st.markdown(
        '<div class="smallcap">One per line, or paste a JSON list like <code>["chunk1","chunk2"]</code>.</div>',
        unsafe_allow_html=True
    )
    st.text_area(
        "Contexts",
        height=330,
        key="contexts_raw",
        placeholder='Context chunk 1...\nContext chunk 2...\n\nOR JSON:\n["chunk1", "chunk2"]'
    )
    st.markdown("</div>", unsafe_allow_html=True)

contexts = parse_contexts(st.session_state.contexts_raw)

action_left, action_right = st.columns([1, 1])
with action_left:
    evaluate = st.button("Evaluate", type="primary")
with action_right:
    st.markdown(
        f'<span class="pill">Contexts: {len(contexts)}</span>'
        f'<span class="pill">Threshold: {sent_support_threshold:.2f}</span>'
        f'<span class="pill">Costs: in={in_cost:.4f}, out={out_cost:.4f}</span>',
        unsafe_allow_html=True
    )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

if evaluate:
    user_query = (st.session_state.user_query or "").strip()
    ai_answer = (st.session_state.ai_answer or "").strip()

    if not user_query:
        st.error("Please enter a User query.")
        st.stop()
    if not ai_answer:
        st.error("Please enter an AI answer.")
        st.stop()
    if not contexts:
        st.error("Please paste at least one context chunk.")
        st.stop()

    cm = CostModel(in_cost, out_cost)
    report = build_report(user_query, ai_answer, contexts, sent_support_threshold, cm)

    st.session_state.history.insert(0, report)
    st.session_state.history = st.session_state.history[:8]

    st.markdown("### Scores")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Relevance", f"{report['relevance_score']:.3f}")
    m2.metric("Completeness", f"{report['completeness_score']:.3f}")
    m3.metric("Factual support", f"{report['factual_support_score']:.3f}")
    m4.metric("Latency (s)", f"{report['latency_seconds']:.4f}")

    st.markdown("### Support breakdown")
    b1, b2 = st.columns(2)
    b1.metric("Lexical support", f"{report['lexical_support_score']:.3f}")
    b2.metric("Numeric penalty", f"{report['numeric_penalty']:.3f}")

    st.markdown("### Cost & Tokens")
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated input tokens", str(report["estimated_tokens"]["input"]))
    c2.metric("Estimated output tokens", str(report["estimated_tokens"]["output"]))
    c3.metric("Estimated cost (USD)", f"{report['estimated_cost_usd']:.6f}")

    tab1, tab2, tab3 = st.tabs(["Potential issues", "Raw JSON", "History"])

    with tab1:
        st.markdown("#### Sentence-level support")
        if report["potentially_unsupported_sentences"]:
            st.warning(
                f"{len(report['potentially_unsupported_sentences'])} sentence(s) look unsupported "
                f"at threshold {sent_support_threshold:.2f}."
            )
            for i, item in enumerate(report["potentially_unsupported_sentences"], start=1):
                st.markdown(f"**{i}.** {item['sentence']}")
                st.caption(f"overlap_ratio={item['overlap_ratio']} | overlap_terms={item['overlap_terms']}")
                st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        else:
            st.success("No unsupported sentences flagged (with current threshold).")

        st.markdown("#### Numeric hallucinations")
        if report["unsupported_numeric_claims"]:
            st.warning("Unsupported numeric claims (not found in retrieved context):")
            st.write(report["unsupported_numeric_claims"])
        else:
            st.success("No unsupported numeric claims flagged.")

    with tab2:
        st.download_button(
            label="Download JSON report",
            data=json.dumps(report, indent=2, ensure_ascii=False),
            file_name="llm_eval_report.json",
            mime="application/json",
        )
        st.code(json.dumps(report, indent=2, ensure_ascii=False), language="json")

    with tab3:
        if not st.session_state.history:
            st.info("No history yet. Run an evaluation first.")
        else:
            for idx, r in enumerate(st.session_state.history, start=1):
                title = (
                    f"Run #{idx} — rel={r['relevance_score']:.3f}, comp={r['completeness_score']:.3f}, "
                    f"factual={r['factual_support_score']:.3f}, latency={r['latency_seconds']:.4f}s"
                )
                with st.expander(title):
                    st.code(json.dumps(r, indent=2, ensure_ascii=False), language="json")
else:
    st.info("Fill the fields above and click Evaluate. Use the sidebar Quick test cases to auto-fill examples.")
