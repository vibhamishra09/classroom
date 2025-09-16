from typing import List, Dict, Any
from collections import Counter
import re
import os

# --- BERTopic for topic extraction (replaces YAKE) ---
try:
    from bertopic import BERTopic
    _HAS_BERTOPIC = True
except Exception:
    BERTopic = None
    _HAS_BERTOPIC = False

# ===== Basic NLP =====
SENTENCE_END = re.compile(r"[.!?]+\s+")
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

STOPWORDS = set(
    "the of and to a in that is it for on with as are was be by this an or from at not your you we they he she i".split()
)

EXAMPLE_MARKERS = re.compile(r"\b(for example|for instance|e\.g\.|let's say|imagine|suppose)\b", re.IGNORECASE)
DEFINITION_MARKERS = re.compile(r"\b(defined as|definition|refers to|means|can be defined as|is called)\b", re.IGNORECASE)
RECAP_MARKERS = re.compile(r"\b(to recap|in summary|to summarize|in conclusion|let's recap|summary)\b", re.IGNORECASE)
CONTRAST_MARKERS = re.compile(r"\b(vs\.?|versus|difference between|compare|contrast)\b", re.IGNORECASE)
MISCON_MARKERS = re.compile(r"\b(misconception|common mistake|people think|not to be confused with)\b", re.IGNORECASE)
QUESTION_WORDS = re.compile(r"\b(what|why|how|when|where|which|who|does|do|is|are|can|should|could|would)\b.*\?", re.IGNORECASE)

FILLERS = {"um", "uh", "like", "so", "basically", "actually", "right", "okay"}


def words(text: str):
    return WORD_RE.findall(text.lower())


def compute_wpm(text: str, duration_sec: float) -> float:
    w = len(words(text))
    minutes = max(duration_sec / 60.0, 1e-6)
    return round(w / minutes, 1)


def filler_stats(text: str) -> Dict[str, Any]:
    tokens = words(text)
    c = Counter(tokens)
    total = sum(c.values()) or 1
    fillers_count = {f: c[f] for f in FILLERS if f in c}
    return {
        "total_words": total,
        "total_fillers": sum(fillers_count.values()),
        "filler_breakdown": fillers_count,
        "filler_ratio_pct": round(100.0 * sum(fillers_count.values()) / total, 2),
    }


def sectioning(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Naive sectioning: new section if pause > 2.5s or a long segment."""
    secs = []
    current = {"start": None, "end": None, "texts": []}
    prev_end = 0.0
    for i, seg in enumerate(segments):
        gap = seg["start"] - prev_end if i else 0.0
        if (gap > 2.5) or (len(seg["text"]) > 200):
            if current["texts"]:
                current["end"] = prev_end
                current["text"] = " ".join(current["texts"]).strip()
                secs.append(current)
            current = {"start": seg["start"], "end": None, "texts": [seg["text"]]}
        else:
            if current["start"] is None:
                current["start"] = seg["start"]
            current["texts"].append(seg["text"])
        prev_end = seg["end"]
    if current["texts"]:
        current["end"] = prev_end
        current["text"] = " ".join(current["texts"]).strip()
        secs.append(current)
    return secs


def structure_score(text: str, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    signposts = [
        "first", "second", "third", "conclusion", "summary", "in this section", "let's recap", "to summarize"
    ]
    tl = text.lower()
    sp_hits = sum(tl.count(s) for s in signposts)
    n = len(sections)
    score = 5
    if 3 <= n <= 9:
        score += 2
    if sp_hits >= 2:
        score += 2
    if len(text) > 500:
        score += 1
    return {"score_out_of_10": min(score, 10), "sections": sections, "signposts": sp_hits}


def jargon_density(text: str) -> float:
    toks = [t for t in words(text) if t not in STOPWORDS]
    if not toks:
        return 0.0
    long = sum(1 for t in toks if len(t) >= 9)
    return round(long / len(toks), 3)


def per_minute(count: int, duration_sec: float) -> float:
    minutes = max(duration_sec / 60.0, 1e-6)
    return round(count / minutes, 2)


# -------- BERTopic replacement for extract_topics --------
_topic_model = None
def _get_topic_model():
    """
    Lazily build a BERTopic model suitable for single-document inputs.
    You can override the embedding model via:
      export TOPIC_EMBED_MODEL=all-MiniLM-L6-v2
    """
    global _topic_model
    if _topic_model is not None:
        return _topic_model

    if not _HAS_BERTOPIC:
        _topic_model = ("__FAILED__", "BERTopic not installed")
        return _topic_model

    try:
        # Build with lenient clustering so a single doc forms a topic
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer

        embed_name = os.getenv("TOPIC_EMBED_MODEL", "all-MiniLM-L6-v2")
        vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
        hdb = HDBSCAN(min_cluster_size=1, min_samples=1)  # allow 1-doc topics

        _topic_model = BERTopic(
            embedding_model=embed_name,
            vectorizer_model=vectorizer,
            hdbscan_model=hdb,
            calculate_probabilities=False,
            verbose=False,
        )
    except Exception as e:
        _topic_model = ("__FAILED__", f"{type(e).__name__}: {e}")
    return _topic_model


def _fallback_keywords(text: str, top_k: int) -> List[str]:
    # Simple, fast backup if BERTopic is unavailable
    toks = [t for t in words(text) if t not in STOPWORDS and len(t) > 2]
    counts = Counter(toks)
    return [w for w, _ in counts.most_common(top_k)]


def extract_topics(text: str, top_k: int = 8) -> list:
    """
    Rewritten to use BERTopic. Returns top_k representative words for
    the dominant topic of this transcript. Falls back to a simple
    frequency-based method if BERTopic fails/is unavailable.
    """
    mdl = _get_topic_model()
    if isinstance(mdl, tuple) and mdl[0] == "__FAILED__":
        return _fallback_keywords(text, top_k)

    try:
        topics, _ = mdl.fit_transform([text])
        topic_id = topics[0]
        # With our HDBSCAN settings, we should get topic_id >= 0.
        # Still, guard and fall back if representation is missing.
        rep = mdl.get_topic(topic_id) or []
        if not rep:
            return _fallback_keywords(text, top_k)
        return [w for (w, _score) in rep[:top_k]]
    except Exception:
        return _fallback_keywords(text, top_k)



def teaching_feedback(text: str, sections: List[Dict[str, Any]], duration_sec: float) -> Dict[str, Any]:
    tl = text.lower()

    # Counts / proxies
    q_marks = text.count("?")
    q_sentences = len(QUESTION_WORDS.findall(text))
    questions = max(q_marks, q_sentences)
    examples = len(EXAMPLE_MARKERS.findall(tl))
    definitions = len(DEFINITION_MARKERS.findall(tl))
    recaps = len(RECAP_MARKERS.findall(tl))
    contrasts = len(CONTRAST_MARKERS.findall(tl))
    misconceptions = len(MISCON_MARKERS.findall(tl))

    jp = jargon_density(text)

    # Supporting metrics
    pacing_wpm = compute_wpm(text, duration_sec)
    sp = structure_score(text, sections)

    # Rubric 1–5
    def score_pacing(wpm: float):
        if 120 <= wpm <= 160:
            return 5
        if 105 <= wpm < 120 or 160 < wpm <= 175:
            return 4
        if 90 <= wpm < 105 or 175 < wpm <= 190:
            return 3
        if 75 <= wpm < 90 or 190 < wpm <= 205:
            return 2
        return 1

    def scale(val, cutpoints):
        # cutpoints ascending → scores 1..5
        for i, cp in enumerate(cutpoints, start=1):
            if val < cp:
                return i
        return 5

    rubric = {
        "clarity": {"score": scale(1 - jp, [0.75, 0.82, 0.88, 0.93]), "why": f"jargon density {jp}"},
        "examples_analogies": {"score": scale(examples, [1, 2, 3, 5]), "why": f"{examples} example markers"},
        "engagement_questions": {"score": scale(per_minute(questions, duration_sec), [0.15, 0.4, 0.8, 1.2]), "why": f"{questions} questions"},
        "definitions_terminology": {"score": scale(definitions, [0, 1, 2, 3]), "why": f"{definitions} definition cues"},
        "structure_signposting": {"score": scale(sp["signposts"], [0, 1, 2, 3]), "why": f"{sp['signposts']} signposts"},
        "pacing": {"score": score_pacing(pacing_wpm), "why": f"{pacing_wpm} WPM"},
        "contrast_misconceptions": {"score": scale(contrasts + misconceptions, [0, 1, 2, 3]), "why": f"{contrasts} contrast + {misconceptions} misconceptions"},
        "recap": {"score": 5 if recaps >= 1 else 2, "why": "recap found" if recaps else "no recap"},
    }

    # Quick wins
    quick_wins = []
    if examples < 2:
        quick_wins.append("Add 2 short, concrete examples. Use numbers or a real scenario.")
    if per_minute(questions, duration_sec) < 0.8:
        quick_wins.append("Ask a check-for-understanding question every ~60–90 seconds.")
    if definitions < 1 or jp > 0.12:
        quick_wins.append("Define key terms early. Keep the first definition to one sentence.")
    if sp["signposts"] < 2:
        quick_wins.append("Add clear signposts: 'First…', 'Next…', then 'To recap…'.")
    if recaps < 1:
        quick_wins.append("End with a 2–3 sentence recap and 1 actionable takeaway.")
    if contrasts + misconceptions < 1:
        quick_wins.append("Briefly contrast with a close concept or address a common misconception.")

    counts = {
        "questions": questions,
        "examples": examples,
        "definitions": definitions,
        "recaps": recaps,
        "contrasts": contrasts,
        "misconceptions": misconceptions,
        "jargon_density": jp,
    }

    return {"rubric": rubric, "quick_wins": quick_wins, "counts": counts}


def analyze_transcript(text: str, segments: List[Dict[str, Any]], duration_sec: float) -> Dict[str, Any]:
    wpm = compute_wpm(text, duration_sec)
    fillers = filler_stats(text)
    secs = sectioning(segments)
    structure = structure_score(text, secs)

    sentences = [s.strip() for s in SENTENCE_END.split(text) if s.strip()]
    avg_sent_len = round(sum(len(words(s)) for s in sentences) / max(len(sentences), 1), 1)

    guidance = []
    if wpm > 165:
        guidance.append("Pacing is fast; aim for 120–160 WPM for lectures.")
    elif wpm < 110:
        guidance.append("Pacing is slow; consider 120–160 WPM.")
    if fillers["filler_ratio_pct"] > 2.0:
        guidance.append("Reduce filler words; pause briefly instead of saying 'um/like'.")
    if structure["score_out_of_10"] < 7:
        guidance.append("Use signposts: 'First…', 'Next…', and end with a recap.")
    if avg_sent_len > 28:
        guidance.append("Shorten sentences for clarity; target 14–22 words on average.")

    teaching = teaching_feedback(text, secs, duration_sec)

    return {
        "wpm": wpm,
        "fillers": fillers,
        "structure": structure,
        "avg_sentence_length_words": avg_sent_len,
        "actionable_advice": guidance,
        "teaching": teaching,
    }


def feedback_to_paragraph(topics, feedback):
    """Turn numeric feedback into a readable paragraph for the UI."""
    try:
        wpm = feedback.get("wpm") or 0
        fillers_pct = (feedback.get("fillers") or {}).get("filler_ratio_pct") or 0
        struct = (feedback.get("structure") or {})
        struct_score = struct.get("score_out_of_10") or 0
        signposts = struct.get("signposts") or 0
        avg_len = feedback.get("avg_sentence_length_words") or 0
        teaching = (feedback.get("teaching") or {})
        quick = teaching.get("quick_wins") or []

        band = "within" if 120 <= wpm <= 160 else ("slightly outside" if 105 <= wpm <= 175 else "outside")
        topic_str = ", ".join(topics[:6]) if topics else "the topic"

        parts = []
        parts.append(
            f"This lecture on {topic_str} averaged {wpm} words per minute, which is {band} the recommended 120–160 WPM range. "
            f"Filler words accounted for {fillers_pct}% of the transcript. The overall structure scored {struct_score}/10 "
            f"with {signposts} clear signpost(s), and the average sentence length was about {avg_len} words."
        )

        if wpm < 110:
            parts.append(" Consider increasing the pace slightly and using shorter phrasing to maintain energy.")
        elif wpm > 165:
            parts.append(" Consider slowing down and pausing at key transitions to aid comprehension.")
        if fillers_pct > 2.0:
            parts.append(" Try replacing filler words with a one-beat pause before key terms.")
        if struct_score < 7:
            parts.append(" Add explicit signposts (e.g., 'First…', 'Next…', 'To recap…') to strengthen structure.")
        if avg_len > 28:
            parts.append(" Shorten dense sentences; aiming for 14–22 words improves clarity.")

        if quick:
            qw = " ".join(q if q.endswith(".") else q + "." for q in quick[:4])
            parts.append(" Quick wins: " + qw)

        return "".join(parts).strip()
    except Exception:
        return "We generated numeric feedback, but formatting it into a paragraph failed. Please check the JSON tab."
