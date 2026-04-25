"""
Natural Language Processing module.
Analyses transcript text segment by segment using FinBERT,
so we get a sentiment score at each point in the call rather
than a single aggregate -- this powers the intra-call timeline.
"""

from transformers import pipeline
import spacy
import numpy as np
import streamlit as st
from speech import is_management_speaker

@st.cache_resource(show_spinner=False)
def _load_finbert():
    """Load FinBERT once per server session via st.cache_resource."""
    try:
        model = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
        return model, True
    except Exception as e:
        print(f"FinBERT unavailable: {e} -- sentiment will default to neutral")
        return None, False


@st.cache_resource(show_spinner=False)
def _load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return None


_finbert, _finbert_available = _load_finbert()
_nlp = _load_spacy()


def _score_text(text):
    """
    Run a single piece of text through FinBERT.
    Returns a net sentiment score in [-1, +1].
    Computed as (positive_prob - negative_prob) so neutral text scores near zero.
    Returns 0.0 if FinBERT is unavailable.
    """
    if not _finbert_available or not text or not text.strip():
        return 0.0
    truncated = text[:1500]
    results = _finbert(truncated)[0]
    scores = {r["label"]: r["score"] for r in results}
    return scores.get("positive", 0.0) - scores.get("negative", 0.0)


def analyse_sentiment(text):
    """
    Analyse a single block of text.
    Used when we have no segment timestamps (transcript-only fallback).
    Returns a dict with net score and individual class probabilities.
    Returns neutral defaults if FinBERT is unavailable.
    """
    if not _finbert_available or not text or not text.strip():
        return {"score": 0.0, "positive": 0.0, "neutral": 1.0, "negative": 0.0}
    truncated = text[:1500]
    results = _finbert(truncated)[0]
    scores = {r["label"]: r["score"] for r in results}
    return {
        "score": scores.get("positive", 0.0) - scores.get("negative", 0.0),
        "positive": scores.get("positive", 0.0),
        "neutral": scores.get("neutral", 0.0),
        "negative": scores.get("negative", 0.0),
    }


def weighted_segment_mean(segments, field="sentiment_score", min_words=8):
    """
    Word-count-weighted mean of a numeric field across enriched segments.

    Segments shorter than min_words are excluded from the average so that
    filler phrases ("Thank you", "Good morning", "Sure") don't dilute the
    signal from information-rich sentences about operations or guidance.
    Longer segments receive proportionally more weight.

    Falls back to an unweighted mean if no segment meets the word threshold
    (e.g. very short audio clips where all segments are brief).
    """
    qualifying = [
        (s.get(field, 0.0), len(s.get("text", "").split()))
        for s in segments
        if len(s.get("text", "").split()) >= min_words
    ]
    if not qualifying:
        vals = [s.get(field, 0.0) for s in segments]
        return float(np.mean(vals)) if vals else 0.0
    total_w = sum(w for _, w in qualifying)
    if total_w == 0:
        return 0.0
    return round(sum(v * w for v, w in qualifying) / total_w, 4)

@st.cache_data(show_spinner=False)
def analyse_segments(mapped_segments, batch_size=16):
    """
    Core function -- run FinBERT over Whisper segments in batches.
    Batching is ~10x faster than one segment at a time on CPU.
    mapped_segments: list of dicts from speech.map_speakers(), each containing
    {start, end, text, speaker}. Returns enriched list adding sentiment fields.
    This segment-level output powers the intra-call timeline chart.
    """
    if not mapped_segments:
        return []

    if not _finbert_available:
        # FinBERT unavailable -- return neutral scores for all segments
        return [{**seg, "sentiment_score": 0.0, "positive": 0.0, "neutral": 1.0, "negative": 0.0}
                for seg in mapped_segments]

    texts = [seg.get("text", "") or "" for seg in mapped_segments]
    all_scores = []

    # Process in batches to avoid memory issues and speed up inference
    for i in range(0, len(texts), batch_size):
        batch = [t[:1500] if t.strip() else "neutral" for t in texts[i:i + batch_size]]
        results = _finbert(batch)
        for res in results:
            scores = {r["label"]: r["score"] for r in res}
            all_scores.append({
                "sentiment_score": scores.get("positive", 0.0) - scores.get("negative", 0.0),
                "positive":        scores.get("positive", 0.0),
                "neutral":         scores.get("neutral",  1.0),
                "negative":        scores.get("negative", 0.0),
            })

    return [{**seg, **score} for seg, score in zip(mapped_segments, all_scores)]


def get_hedging_frequency(segments):
    """
    Count hedging language across all segments.
    Hedging words signal management caution -- an increase vs. prior quarters
    is a risk flag. Returns raw count and frequency per 100 words.
    """
    HEDGE_WORDS = {
        "cautious", "uncertain", "uncertainty", "headwinds", "challenging",
        "difficult", "volatile", "risk", "risks", "concern", "concerns",
        "potentially", "may", "might", "could", "subject to", "dependent",
        "depends", "if", "assuming", "approximately", "roughly", "around",
        "guidance", "visibility", "limited", "pressured", "pressure"
    }
    total_words = 0
    hedge_count = 0
    for seg in segments:
        words = seg.get("text", "").lower().split()
        total_words += len(words)
        hedge_count += sum(1 for w in words if w.strip(".,;:") in HEDGE_WORDS)
    frequency = (hedge_count / total_words * 100) if total_words > 0 else 0.0
    return {"count": hedge_count, "frequency_per_100": round(frequency, 2)}


_FINANCIAL_KEYWORDS = {
    "revenue", "earnings", "profit", "growth", "margin", "guidance", "outlook",
    "forecast", "billion", "million", "eps", "ebitda", "cash", "dividend",
    "buyback", "demand", "supply", "cost", "expense", "investment", "acquisition",
    "customer", "product", "market", "quarter", "annual", "sales", "operating",
    "gross", "net", "income", "loss", "share", "return", "beat", "miss",
}


def extract_talking_points(segments=None, transcript_text=None, n=6):
    """
    Extract key talking points ranked by financial keyword density and sentiment extremity.
    Prefers enriched segments (which carry timestamps); falls back to raw transcript text.
    Returns a list of dicts: {text, sentiment, time_min (or None)}.
    """
    import re

    if segments:
        candidates = [
            {
                "text": s.get("text", "").strip(),
                "sentiment": s.get("sentiment_score", 0.0),
                "time_min": round(s.get("start", 0) / 60, 1),
            }
            for s in segments if len(s.get("text", "").strip()) > 25
        ]
    elif transcript_text:
        sentences = re.split(r"(?<=[.!?])\s+", transcript_text)
        candidates = [
            {"text": s.strip(), "sentiment": _score_text(s), "time_min": None}
            for s in sentences if len(s.strip()) > 25
        ]
    else:
        return []

    def rank(c):
        words = c["text"].lower().split()
        kw_hits = sum(1 for w in words if w.strip(".,;:") in _FINANCIAL_KEYWORDS)
        return kw_hits * 2 + abs(c["sentiment"])

    return sorted(candidates, key=rank, reverse=True)[:n]


def analyse_transcript_text(transcript_text):
    """
    Quick multi-sample analysis of a raw transcript string for QoQ comparison.
    Samples 5 evenly-spaced windows of 1500 chars across the full transcript
    and averages the FinBERT scores — avoids the boilerplate-truncation bias
    that occurs when only the opening 1500 chars (safe harbour / operator intro)
    are scored.
    Returns a dict with overall sentiment and hedging frequency.
    """
    if not transcript_text or not transcript_text.strip():
        return {"sentiment": 0.0, "positive": 0.0, "negative": 0.0, "hedge_freq": 0.0}

    # Sample evenly across the transcript; skip the first 10% (boilerplate)
    n_samples  = 5
    chunk_size = 1500
    text       = transcript_text.strip()
    start_off  = max(0, int(len(text) * 0.10))  # skip opening boilerplate
    usable_len = len(text) - start_off
    step       = max(chunk_size, usable_len // n_samples)

    scores_pos, scores_neg, scores_neu = [], [], []
    for i in range(n_samples):
        offset  = start_off + i * step
        if offset >= len(text):
            break
        chunk   = text[offset: offset + chunk_size]
        result  = analyse_sentiment(chunk)
        scores_pos.append(result["positive"])
        scores_neg.append(result["negative"])
        scores_neu.append(result["neutral"])

    if not scores_pos:
        return {"sentiment": 0.0, "positive": 0.0, "negative": 0.0, "hedge_freq": 0.0}

    avg_pos  = float(np.mean(scores_pos))
    avg_neg  = float(np.mean(scores_neg))
    avg_neu  = float(np.mean(scores_neu))
    avg_sent = round(avg_pos - avg_neg, 4)

    fake_seg = [{"text": transcript_text}]
    hedge    = get_hedging_frequency(fake_seg)
    return {
        "sentiment":  avg_sent,
        "positive":   round(avg_pos, 4),
        "negative":   round(avg_neg, 4),
        "hedge_freq": hedge["frequency_per_100"],
    }


def parse_av_speakers(av_transcript):
    """
    Takes the JSON list from fetch_transcript_cached and 
    splits it into the two structures needed for resolution.
    """
    av_turns = []
    title_map = {}

    for item in av_transcript:
        name = item.get("speaker", "Unknown")
        title = item.get("title", "")
        content = item.get("content", "")

        if name:
            # Populate the title map
            if title:
                title_map[name] = title
            
            # Populate the turns for word-overlap matching
            av_turns.append({
                "name": name, 
                "text": content
            })

    return av_turns, title_map


def compute_nsi(current_stats, historical_stats):
    """
    Narrative Shift Index -- measures how far the current call's language
    has shifted relative to the previous N quarters.
    current_stats: dict with sentiment, hedge_freq, positive, negative keys.
    historical_stats: list of similar dicts (oldest first).
    Returns a dict with nsi_sigma (z-score), direction, and component deltas.
    A positive sigma means unusually positive vs history; negative = unusually cautious.
    """
    if not historical_stats:
        return {"nsi_sigma": 0.0, "direction": "Insufficient history", "delta_sentiment": 0.0, "delta_hedge": 0.0}

    hist_sentiments = [h["sentiment"] for h in historical_stats if h.get("sentiment") is not None]
    hist_hedges     = [h["hedge_freq"] for h in historical_stats if h.get("hedge_freq") is not None]

    if len(hist_sentiments) < 2:
        return {"nsi_sigma": 0.0, "direction": "Insufficient history", "delta_sentiment": 0.0, "delta_hedge": 0.0}

    hist_mean = float(np.mean(hist_sentiments))
    hist_std  = float(np.std(hist_sentiments))

    cur_sent  = current_stats.get("sentiment", 0.0)
    cur_hedge = current_stats.get("hedge_freq", 0.0)
    hist_hedge_mean = float(np.mean(hist_hedges)) if hist_hedges else 0.0

    hist_std_floored = max(hist_std, 0.05)  # floor: prevent absurd sigma from near-zero variance
    nsi_sigma = round((cur_sent - hist_mean) / hist_std_floored, 2) if hist_std > 0 else 0.0

    if nsi_sigma > 1.0:
        direction = "Meaningfully more positive"
    elif nsi_sigma > 0.5:
        direction = "Slightly more positive"
    elif nsi_sigma < -1.0:
        direction = "Meaningfully more cautious"
    elif nsi_sigma < -0.5:
        direction = "Slightly more cautious"
    else:
        direction = "Consistent with history"

    return {
        "nsi_sigma":        nsi_sigma,
        "direction":        direction,
        "delta_sentiment":  round(cur_sent - hist_mean, 3),
        "delta_hedge":      round(cur_hedge - hist_hedge_mean, 2),
        "hist_mean":        round(hist_mean, 3),
        "hist_std":         round(hist_std, 3),
        "n_quarters":       len(hist_sentiments),
    }


_QA_TRANSITION_PHRASES = [
    "we will now begin the question",
    "we will now open the floor",
    "we'll now open the floor",
    "open the floor to questions",
    "open for questions",
    "now open for questions",
    "operator, please open",
    "your first question",
    "first question comes from",
    "first question is from",
    "we'll take our first question",
    "take the first question",
    "question-and-answer session",
    "question and answer session",
    "q&a session",
    "begin q&a",
    "start the q&a",
    "open the q&a",
    "ready for questions",
    "first question comes from",
    "first question is from",
    "next question comes from",
    "our first question",
]


def find_qa_start_time(segments):
    """
    Q&A Detection Logic:
    1. Priority: First Resolved Analyst (post 10-min mark).
    2. Secondary: Operator Transition Phrases (interpolated).
    3. Fallback: None (Caller defaults to 65% heuristic).
    """
    _OPERATOR_LABEL = "operator"
    _MIN_QA_SECONDS  = 600  # 10 Minutes

    # PRE-SORT: Ensure we are checking chronologically
    sorted_segs = sorted(segments, key=lambda x: x.get("start", 0))

    # --- STRATEGY 1: THE ANALYST TRIGGER (HIGHEST CONFIDENCE) ---
    # We look for the first person who is NOT Management, NOT an Operator,
    # and NOT an unresolved 'SPEAKER_XX' label.
    for seg in sorted_segs:
        start = float(seg.get("start", 0))
        if start < _MIN_QA_SECONDS:
            continue
            
        speaker = seg.get("speaker", "")
        # is_management_speaker returns False specifically for Analysts
        if is_management_speaker(speaker) is False:
            # Final check to ensure it's not the Operator/Moderator
            if not any(lbl in speaker.lower() for lbl in _OPERATOR_LABEL):
                return start

    # --- STRATEGY 2: PHRASE-BASED INTERPOLATION (BACKUP) ---
    for seg in sorted_segs:
        start = float(seg.get("start", 0))
        if start < _MIN_QA_SECONDS:
            continue

        text_lower = seg.get("text", "").lower()
        end = float(seg.get("end", start + 10))
        
        for phrase in _QA_TRANSITION_PHRASES:
            if phrase in text_lower:
                # If Operator says it, QA starts AFTER the segment
                if any(lbl in seg.get("speaker", "").lower() for lbl in _OPERATOR_LABEL):
                    return end
                
                # If Management says it, interpolate the exact second
                phrase_pos = text_lower.find(phrase)
                frac = phrase_pos / max(len(text_lower), 1)
                return round(start + frac * (end - start), 2)

    return None # Pipeline will apply 65% heuristic


def split_prepared_vs_qa(segments, search_segments=None):
    """
    Split segments into prepared remarks and Q&A.

    search_segments: the segment list to scan for Q&A transition phrases.
        Pass all enriched_segments here (including operator turns) so the
        operator's announcement is not missed when `segments` has been
        pre-filtered to management-only.  Defaults to `segments` when omitted.

    Falls back to the 65% duration heuristic only when no phrase is found.
    Price et al. (2012) identify the Q&A section as the highest-alpha section.
    Returns two lists: (prepared_segments, qa_segments)
    """
    if not segments:
        return [], []

    source = search_segments if search_segments is not None else segments
    qa_start_time = find_qa_start_time(source)

    if qa_start_time is not None:
        prepared = [s for s in segments if s.get("start", 0) < qa_start_time]
        qa       = [s for s in segments if s.get("start", 0) >= qa_start_time]
        return prepared, qa

    # Fallback: 65% duration heuristic
    max_time = max(seg.get("end", 0) for seg in segments)
    qa_start_time = max_time * 0.65
    prepared = [s for s in segments if s.get("start", 0) < qa_start_time]
    qa       = [s for s in segments if s.get("start", 0) >= qa_start_time]
    return prepared, qa


def compute_text_qa_stress(text):
    """
    Compute Q&A stress from a plain transcript string (no timestamps).
    Splits at the first Q&A transition phrase; falls back to 65% position.
    Returns prepared_sentiment - qa_sentiment (positive = stress in Q&A).
    Returns 0.0 if the text is too short to split meaningfully.
    """
    if not text or len(text) < 500:
        return 0.0

    split_pos = None
    text_lower = text.lower()
    for phrase in _QA_TRANSITION_PHRASES:
        idx = text_lower.find(phrase)
        if idx > 0:
            split_pos = idx
            break

    if split_pos is None:
        split_pos = int(len(text) * 0.65)

    prepared_text = text[:split_pos].strip()
    qa_text       = text[split_pos:].strip()

    if len(prepared_text) < 200 or len(qa_text) < 200:
        return 0.0

    prep_sent = analyse_sentiment(prepared_text[:1500])["score"]
    qa_sent   = analyse_sentiment(qa_text[:1500])["score"]
    return round(prep_sent - qa_sent, 3)


# Common words to exclude from keyword extraction (financial context aware)
_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "that", "this", "these", "those", "it",
    "its", "we", "our", "us", "i", "you", "your", "they", "their",
    "he", "she", "his", "her", "not", "no", "as", "by", "from", "so",
    "if", "when", "what", "which", "who", "how", "all", "more", "than",
    "about", "up", "out", "into", "over", "then", "also", "very", "just",
    "other", "can", "well", "some", "there", "one", "two", "three",
    "first", "second", "quarter", "year", "q1", "q2", "q3", "q4",
    "fiscal", "thank", "thanks", "think", "know", "see", "going", "get",
    "good", "great", "really", "much", "now", "next", "last", "like",
}


def extract_key_insights(segments, n=3):
    """
    Extract the most analytically significant moments from the call.
    Uses per-segment FinBERT scores already computed in analyse_segments().
    Returns three lists: management highlights, risk signals, hedging moments.
    These power the Key Insights panel in the dashboard.
    """
    if not segments:
        return {"highlights": [], "risk_signals": [], "hedging_moments": []}

    valid = [s for s in segments if s.get("text", "").strip()]

    def _speaker_entry(s):
        """Return speaker name from segment, already includes title if resolved."""
        return s.get("speaker", "") or ""

    # Top positive moments -- management expressing confidence or strong results
    highlights = sorted(valid, key=lambda s: s.get("sentiment_score", 0), reverse=True)
    highlights = [
        {
            "text":     s["text"].strip(),
            "score":    round(s.get("sentiment_score", 0), 2),
            "time_min": round(s.get("start", 0) / 60, 1),
            "speaker":  _speaker_entry(s),
        }
        for s in highlights[:n] if s.get("sentiment_score", 0) > 0.05
    ]

    # Top negative moments -- stress, disappointment, or cautious framing
    risk_signals = sorted(valid, key=lambda s: s.get("sentiment_score", 0))
    risk_signals = [
        {
            "text":     s["text"].strip(),
            "score":    round(s.get("sentiment_score", 0), 2),
            "time_min": round(s.get("start", 0) / 60, 1),
            "speaker":  _speaker_entry(s),
        }
        for s in risk_signals[:n] if s.get("sentiment_score", 0) < -0.05
    ]

    return {
        "highlights":   highlights,
        "risk_signals": risk_signals,
    }


def get_top_keywords(segments, transcript_text=None, n=15):
    """
    Extract top keywords by word frequency from enriched segments or a raw transcript.
    Uses segments when available (preferred); falls back to transcript_text.
    Returns a list of (word, count) tuples sorted by count descending.
    """
    if segments:
        text = " ".join(seg.get("text", "") for seg in segments)
    elif transcript_text:
        text = transcript_text
    else:
        return []

    counts = {}
    for w in text.lower().split():
        w = w.strip(".,;:!?\"'()[]--")
        if len(w) < 3 or w in _STOP_WORDS or not w.isalpha():
            continue
        counts[w] = counts.get(w, 0) + 1

    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]
