#!/usr/bin/env python3
"""BeyondChats - LLM Response Reliability Evaluation Pipeline

Input:
  1) chat conversation JSON (may be slightly non-strict JSON in provided samples)
  2) context vectors JSON from vector DB

Outputs:
  - JSON report with relevance/completeness, hallucination/factuality heuristics,
    and latency/cost estimates.

Notes:
  - Designed to run fast (no external API calls).
  - Metrics are heuristic but production-pluggable.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Robust-ish parsing helpers
# -------------------------

def _strip_line_comments(text: str) -> str:
    return re.sub(r"//.*", "", text)


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _fix_malformed_id_lines(text: str) -> str:
    text = re.sub(r'("id"\s*:\s*\d+)\s*,[^"\n]*"', r"\1,", text)
    return text


def load_context_json(path: str) -> Dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    raw = _strip_line_comments(raw)
    raw = _fix_malformed_id_lines(raw)
    raw = _remove_trailing_commas(raw)
    return json.loads(raw)


def load_chat_json_loose(path: str) -> Dict[str, Any]:
    """Parse chat sample even if it contains comments or wrapped URLs."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    raw = _strip_line_comments(raw)

    chat_id = None
    user_id = None
    m = re.search(r'"chat_id"\s*:\s*(\d+)', raw)
    if m:
        chat_id = int(m.group(1))
    m = re.search(r'"user_id"\s*:\s*(\d+)', raw)
    if m:
        user_id = int(m.group(1))

    turns: List[Dict[str, Any]] = []
    pat = re.compile(
        r'"turn"\s*:\s*(\d+).*?"role"\s*:\s*"([^"]+)".*?'
        r'"message"\s*:\s*"(.*?)"\s*,\s*"created_at"\s*:\s*"([^"]+)"',
        re.DOTALL,
    )

    for m in pat.finditer(raw):
        turn = int(m.group(1))
        role = m.group(2)
        msg = m.group(3)
        created_at = m.group(4)

        msg = msg.replace("\\/", "/").replace('\\"', '"')

        try:
            msg = bytes(msg, "utf-8").decode("unicode_escape")
        except Exception:
            pass

        msg = msg.replace("\\n", "\n")
        msg = msg.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        msg = re.sub(r"\s+", " ", msg).strip()

        for marker in ['"created_at"', '"evaluation_note"', '{ "turn"', '{"turn"', '"turn"']:
            idx = msg.find(marker)
            if idx != -1:
                msg = msg[:idx].strip()

        block_start = m.start()
        block_end = raw.find("}", block_start)
        block = raw[block_start:block_end + 1] if block_end != -1 else raw[block_start:m.end()]
        ev = re.search(r'"evaluation_note"\s*:\s*"(.*?)"', block, re.DOTALL)
        if ev:
            note = ev.group(1).replace("\n", " ")
            note = re.sub(r"\s+", " ", note).strip()
        else:
            note = None

        item: Dict[str, Any] = {
            "turn": turn,
            "role": role,
            "message": msg,
            "created_at": created_at,
        }
        if note:
            item["evaluation_note"] = note
        turns.append(item)

    return {"chat_id": chat_id, "user_id": user_id, "conversation_turns": turns}


# -------------------------
# Metric implementations
# -------------------------

_WORD = re.compile(r"[A-Za-z0-9']+")

# Only "real" numbers (>= 3 digits) or comma-separated numbers.
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
    """Lexical grounding score: sentence overlap with retrieved contexts."""
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
    """Penalty applied to lexical support if numeric claims are unsupported."""
    if not unsupported_nums:
        return 0.0
    # Simple, interpretable penalty:
    # any numeric hallucination reduces factual support by 0.5
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


# -------------------------
# Orchestration
# -------------------------

def choose_target_turns(conversation_turns: List[Dict[str, Any]]) -> List[Tuple[Optional[str], Dict[str, Any]]]:
    pairs: List[Tuple[Optional[str], Dict[str, Any]]] = []
    last_user: Optional[str] = None
    for t in conversation_turns:
        role = (t.get("role") or "").lower()
        if "user" in role and "chatbot" not in role:
            last_user = t.get("message") or ""
        elif "ai" in role or "chatbot" in role:
            if last_user is not None:
                pairs.append((last_user, t))
                last_user = None

    if not pairs:
        for t in reversed(conversation_turns):
            role = (t.get("role") or "").lower()
            if "ai" in role or "chatbot" in role:
                pairs.append((None, t))
                break
    return pairs


def _clean_broken_links(text: str) -> str:
    s = (text or "").strip()
    bad = s.find("](https:")
    if bad != -1 and s.find(")", bad) == -1:
        s = s[:bad].strip()
    if s.endswith("(https:"):
        s = s[:-6].strip()
    return s


def run(chat_path: str, context_path: str, cost_model: CostModel) -> Dict[str, Any]:
    t0 = time.perf_counter()

    chat = load_chat_json_loose(chat_path)
    ctx = load_context_json(context_path)

    vector_data = (ctx.get("data") or {}).get("vector_data") or []
    contexts = [vd.get("text", "") for vd in vector_data if isinstance(vd, dict)]

    pairs = choose_target_turns(chat.get("conversation_turns") or [])
    results = []

    for user_query, ai_turn in pairs:
        response = _clean_broken_links(ai_turn.get("message", ""))

        rel = relevance_score(response, contexts)
        comp = completeness_score(response, user_query or "", contexts) if user_query else None

        lexical_support, flagged = hallucination_score(response, contexts, threshold=0.30)
        unsupported_nums = unsupported_numeric_claims(response, contexts)

        pen = numeric_penalty(unsupported_nums)
        factual_support = max(0.0, min(1.0, lexical_support - pen))

        in_tokens = estimate_tokens((user_query or "") + " " + " ".join(contexts[:3]))
        out_tokens = estimate_tokens(response)
        est_cost = cost_model.estimate(in_tokens, out_tokens)

        results.append({
            "turn": ai_turn.get("turn"),
            "user_query": user_query,
            "assistant_response": response,
            "relevance_score": round(rel, 3),
            "completeness_score": (round(comp, 3) if comp is not None else None),

            # transparent breakdown
            "lexical_support_score": round(lexical_support, 3),
            "numeric_penalty": round(pen, 3),
            "factual_support_score": round(factual_support, 3),

            "potentially_unsupported_sentences": flagged[:10],
            "unsupported_numeric_claims": unsupported_nums,
            "estimated_tokens": {"input": in_tokens, "output": out_tokens},
            "estimated_cost_usd": round(est_cost, 6),
        })

    latency = time.perf_counter() - t0
    return {
        "chat_file": chat_path,
        "context_file": context_path,
        "evaluated_pairs": len(results),
        "latency_seconds": round(latency, 4),
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", required=True, help="Path to chat conversation JSON")
    ap.add_argument("--context", required=True, help="Path to context vectors JSON")
    ap.add_argument("--out", default="", help="Optional path to write JSON report")
    ap.add_argument("--usd_per_1k_input", type=float, default=0.0, help="Cost model: $ per 1k input tokens")
    ap.add_argument("--usd_per_1k_output", type=float, default=0.0, help="Cost model: $ per 1k output tokens")
    args = ap.parse_args()

    cost_model = CostModel(
        usd_per_1k_input_tokens=args.usd_per_1k_input,
        usd_per_1k_output_tokens=args.usd_per_1k_output,
    )

    report = run(args.chat, args.context, cost_model)
    out_text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(out_text, encoding="utf-8")
    print(out_text)


if __name__ == "__main__":
    main()
