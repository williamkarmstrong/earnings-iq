"""
Natural Language Processing module.
Analyses transcript text segment by segment using FinBERT,
so we get a sentiment score at each point in the call rather
than a single aggregate -- this powers the intra-call timeline.
"""

from transformers import pipeline
import spacy
import numpy as np

# Load FinBERT once at module level so it isn't reloaded on every function call.
# FinBERT is fine-tuned on financial text so outperforms general-purpose
# sentiment models on earnings call language.
# Wrapped in try/except -- if the model hasn't downloaded yet the app still runs,
# but sentiment scores will return 0.0 (neutral).
try:
    _finbert = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        top_k=None,
    )
    _finbert_available = True
except Exception as _e:
    print(f"FinBERT unavailable: {_e} -- sentiment will default to neutral")
    _finbert = None
    _finbert_available = False

# spaCy for sentence splitting
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    _nlp = None


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
    Quick single-pass analysis of a raw transcript string for QoQ comparison.
    Returns a dict with overall sentiment and hedging frequency.
    """
    if not transcript_text or not transcript_text.strip():
        return {"sentiment": 0.0, "positive": 0.0, "negative": 0.0, "hedge_freq": 0.0}
    sentiment = analyse_sentiment(transcript_text)
    fake_seg  = [{"text": transcript_text}]
    hedge     = get_hedging_frequency(fake_seg)
    return {
        "sentiment":  sentiment["score"],
        "positive":   sentiment["positive"],
        "negative":   sentiment["negative"],
        "hedge_freq": hedge["frequency_per_100"],
    }


def parse_av_speakers(transcript_text):
    """
    Extract speaker-attributed turns from an Alpha Vantage transcript.
    AV format: lines beginning with "Speaker Name: dialogue..."
    Also attempts to extract titles (CEO, CFO etc.) from introduction lines.
    Returns:
      - turns: list of {name, text} in order
      - title_map: dict of {name: title} where title was found
    """
    import re

    # Match "Name: text" -- name is 2-50 chars, starts with capital, no digits
    turn_re = re.compile(
        r'^([A-Z][A-Za-z\s\.\-]{1,48}):\s+(.+?)(?=\n[A-Z][A-Za-z\s\.\-]{1,48}:|\Z)',
        re.MULTILINE | re.DOTALL,
    )

    # Title patterns in intro lines e.g. "Tim Cook, Chief Executive Officer"
    title_re = re.compile(
        r'([A-Z][A-Za-z\s\.]{2,40}),\s*(Chief Executive Officer|CEO|Chief Financial Officer|CFO|'
        r'Chief Operating Officer|COO|President|Analyst|Operator|Investor Relations)',
        re.IGNORECASE,
    )

    turns = []
    for m in turn_re.finditer(transcript_text):
        name = m.group(1).strip()
        text = m.group(2).strip().replace('\n', ' ')
        if len(text) > 8 and not any(d.isdigit() for d in name):
            turns.append({"name": name, "text": text})

    title_map = {}
    for m in title_re.finditer(transcript_text):
        name  = m.group(1).strip()
        title = m.group(2).strip()
        # Normalise common variants
        title = title.replace("Chief Executive Officer", "CEO").replace(
                              "Chief Financial Officer", "CFO").replace(
                              "Chief Operating Officer", "COO")
        title_map[name] = title

    return turns, title_map


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

    nsi_sigma = round((cur_sent - hist_mean) / hist_std, 2) if hist_std > 0 else 0.0

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


def split_prepared_vs_qa(segments):
    """
    Split segments into prepared remarks and Q&A.
    Whisper doesn't label sections so we use a heuristic: Q&A typically
    starts around 65% through the call duration.
    Price et al. (2012) identify this split as the highest-alpha section.
    Returns two lists: (prepared_segments, qa_segments)
    """
    if not segments:
        return [], []
    max_time = max(seg.get("end", 0) for seg in segments)
    qa_start = max_time * 0.65
    prepared = [s for s in segments if s.get("start", 0) < qa_start]
    qa = [s for s in segments if s.get("start", 0) >= qa_start]
    return prepared, qa


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

    # Top positive moments -- management expressing confidence or strong results
    highlights = sorted(valid, key=lambda s: s.get("sentiment_score", 0), reverse=True)
    highlights = [
        {
            "text": s["text"].strip(),
            "score": round(s.get("sentiment_score", 0), 2),
            "time_min": round(s.get("start", 0) / 60, 1),
        }
        for s in highlights[:n] if s.get("sentiment_score", 0) > 0.05
    ]

    # Top negative moments -- stress, disappointment, or cautious framing
    risk_signals = sorted(valid, key=lambda s: s.get("sentiment_score", 0))
    risk_signals = [
        {
            "text": s["text"].strip(),
            "score": round(s.get("sentiment_score", 0), 2),
            "time_min": round(s.get("start", 0) / 60, 1),
        }
        for s in risk_signals[:n] if s.get("sentiment_score", 0) < -0.05
    ]

    # Hedging language moments -- signals of management caution or uncertainty
    HEDGE_WORDS = {
        "uncertain", "uncertainty", "headwinds", "challenging", "difficult",
        "volatile", "risk", "risks", "concern", "concerns", "potentially",
        "guidance", "limited", "pressured", "pressure", "cautious",
    }
    hedging_moments = []
    for s in valid:
        words = s.get("text", "").lower().split()
        hits = list({w.strip(".,;:") for w in words if w.strip(".,;:") in HEDGE_WORDS})
        if hits:
            hedging_moments.append({
                "text": s["text"].strip(),
                "hedge_words": hits[:3],
                "time_min": round(s.get("start", 0) / 60, 1),
            })

    return {
        "highlights": highlights,
        "risk_signals": risk_signals,
        "hedging_moments": hedging_moments[:n],
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
