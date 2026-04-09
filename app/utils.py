import streamlit as st
import yfinance as yf
import html as _html_lib
import re
from ingestion import fetch_transcript_cached, fetch_backup_transcript
from nlp import analyse_transcript_text, compute_text_qa_stress
from insights import SIGNAL_MCI_POSITIVE, SIGNAL_MCI_WATCH

@st.cache_data(ttl=86400, show_spinner=False)
def _analyse_peer(peer_ticker, period, year):
    """
    Fetch and analyse a peer company's transcript for the given quarter.
    Returns dict with mci, qa_stress, signal — or None on failure.
    Cached 24h so repeat runs use local cache, not API credits.
    """
    try:
        text, err = fetch_transcript_cached(peer_ticker, period, year)
        if not text:
            text, err = fetch_backup_transcript(peer_ticker, period, year)
        if not text:
            return None
        stats    = analyse_transcript_text(text)
        text_mci = round(((stats["sentiment"] + 1) / 2) * 100, 1)
        qa_stress = compute_text_qa_stress(text)
        signal   = "Positive" if text_mci >= SIGNAL_MCI_POSITIVE else "Watch" if text_mci <= SIGNAL_MCI_WATCH else "Neutral"
        return {"mci": text_mci, "qa_stress": qa_stress, "signal": signal}
    except Exception:
        return None

@st.cache_data
def is_valid_ticker(ticker):
    """Quick yfinance check to catch typos before running the pipeline."""
    try:
        hist = yf.Ticker(ticker).history(period="1d")
        return hist.empty is False
    except Exception:
        return False


def get_previous_quarters(period, year, n=2):
    """Return list of (period, year) for the n quarters before the given one."""
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    idx = quarters.index(period)
    result = []
    cur_idx, cur_year = idx, year
    for _ in range(n):
        cur_idx -= 1
        if cur_idx < 0:
            cur_idx = 3
            cur_year -= 1
        result.append((quarters[cur_idx], cur_year))
    return result


def _esc(text):
    """HTML-escape text before injecting into HTML blocks."""
    return _html_lib.escape(str(text)) if text else ""


def _speaker_from_turns(av_turns, text):
    """Match text to best AV transcript speaker via normalised word overlap."""
    if not av_turns or not text:
        return ""
    
    def _nw(t):
        return set(re.sub(r"[^a-z0-9\s]", "", t.lower()).split())
        
    words = _nw(text)
    if len(words) < 2:
        return ""
    best_name, best_score = "", 0
    for turn in av_turns:
        overlap = len(words & _nw(turn["text"]))
        if overlap > best_score:
            best_score = overlap
            best_name = turn["name"]
    return best_name if best_score >= 2 else ""


def sentiment_colour(score):
    """Return a hex colour on a red-to-green gradient for a sentiment score in [-1, +1]."""
    t = (score + 1) / 2          # map [-1,+1] -> [0,1]
    t = max(0.0, min(1.0, t))
    r = int(220 * (1 - t))
    g = int(180 * t)
    return f"#{r:02x}{g:02x}50"  # muted red -> muted green


def _speaker_for_time(segs, time_min, fallback_turns=None, text=""):
    """Return speaker label near time_min; scans 5 nearest segments for a resolved name."""
    if segs:
        nearby = sorted(segs, key=lambda s: abs(s.get("start", 0) / 60 - time_min))[:5]
        for s in nearby:
            sp = s.get("speaker", "")
            if sp and sp not in ("UNKNOWN", ""):
                return sp
    if fallback_turns and text:
        return _speaker_from_turns(fallback_turns, text)
    return ""


def _key_takeaways(flags, nsi, overall_sentiment, hedge_val):
    """Generate 3 plain-English takeaways from signal data."""
    items = []
    high_flags = [f for f in flags if f["severity"] == "high"]
    for f in high_flags[:2]:
        items.append(("high", f["message"]))
    med_flags = [f for f in flags if f["severity"] == "medium"]
    for f in med_flags[:1]:
        items.append(("medium", f["message"]))
    if not items:
        items.append(("low", "No high-priority signals. Tone broadly consistent with text."))
    return items[:3]
