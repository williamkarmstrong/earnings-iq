"""
Streamlit app -- main entry point.

Multimodal sources:
  Audio mode  -> Whisper transcription -> per-segment FinBERT sentiment
                                       -> librosa acoustics + Wav2Vec2 confidence proxy
  Transcript  -> Alpha Vantage (cached locally) -> talking points, QoQ comparison
  Toggle      -> 'Transcript only' sidebar switch bypasses audio for fast UI testing
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
import html as _html_lib

from ingestion import fetch_backup_transcript, fetch_audio, fetch_transcript_cached
from event_study import run_event_study, get_sector_sensitivity_df
from speech import transcribe_audio, map_speakers, resolve_speaker_names
from nlp import (
    analyse_segments, analyse_sentiment, get_hedging_frequency,
    split_prepared_vs_qa, get_top_keywords, extract_key_insights,
    extract_talking_points, analyse_transcript_text, compute_nsi,
    parse_av_speakers, _finbert_available,
)
from audio import extract_audio_features, _wav2vec2_available
from multimodal import analyse_multimodal
from insights import generate_insights

st.set_page_config(page_title="EarningsIQ", layout="wide")


@st.cache_data
def is_valid_ticker(ticker):
    """Quick yfinance check to catch typos before running the pipeline."""
    try:
        return yf.Ticker(ticker).fast_info["lastPrice"] is not None
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
    import re as _re2
    def _nw(t):
        return set(_re2.sub(r"[^a-z0-9\s]", "", t.lower()).split())
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


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### EarningsIQ")
    st.caption("CFA AI Investment Challenge")
    st.divider()

    ticker = st.text_input("Ticker", "AAPL").upper()
    col_p, col_y = st.columns(2)
    period = col_p.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4"])
    year   = col_y.selectbox("Year", list(range(2026, 2017, -1)), index=2)

    transcript_only = st.toggle(
        "Transcript-only mode",
        value=False,
        help="ON: skips audio (Whisper/Wav2Vec2/librosa) -- uses Alpha Vantage transcript only. "
             "Faster for UI testing. OFF (default): full multimodal pipeline.",
    )
    if transcript_only:
        st.caption("Mode: Transcript only")
    else:
        st.caption("Mode: Audio + Text (default)")

    run = st.button("Analyse Call", use_container_width=True, type="primary")
    st.divider()

    with st.expander("Metrics guide"):
        st.markdown("""
**MCI (0-100):** FinBERT 40% + Wav2Vec2 35% + Pause 15% + Pitch 10% *(full multimodal)*

**MCI in peer table:** FinBERT text-only for fair cross-peer comparison

**NSI (sigma):** Sentiment z-score vs prior 6 quarters

**Divergence:** Acoustic vs textual gap scaled to ±0.5 (negative = scripted positivity; flag at −0.12)

**Hedging /100w:** Uncertainty word density (flag threshold: 8.0)

**Q&A Stress:** Sentiment drop from prepared remarks to Q&A (65% split)
        """)

# ============================================================
# MAIN
# ============================================================
st.title("EarningsIQ - Multimodal Earnings Analysis")

if not run:
    st.info("Select a ticker and quarter in the sidebar, then click **Analyse Call**.")
    st.stop()

# Auto-clear cache when the selected call changes -- ensures pyannote/Whisper reruns on new input
_call_key = f"{ticker}_{period}_{year}"
if st.session_state.get("last_call_key") != _call_key:
    st.cache_data.clear()
    st.session_state["last_call_key"] = _call_key

if not is_valid_ticker(ticker):
    st.error(f"'{ticker}' is not a recognised ticker.")
    st.stop()

# ============================================================
# PIPELINE
# ============================================================
progress = st.progress(0)
status   = st.empty()
pipeline_warnings = []

enriched_segments = []
transcript_text   = None
audio_features    = {}
av_turns          = []
title_map         = {}

if transcript_only:
    # --- Transcript-only mode: skip all audio processing ---
    status.text("Fetching Alpha Vantage transcript (cached locally if available)...")
    transcript_text, err = fetch_transcript_cached(ticker, period, year)
    if not transcript_text:
        st.error(f"Could not load transcript: {err}")
        st.stop()
    st.success("Transcript loaded (transcript-only mode -- audio skipped).")
    try:
        av_turns, title_map = parse_av_speakers(transcript_text)
    except Exception:
        pass
    progress.progress(50)
    audio_features = {
        "confidence_proxy": 0.5,
        "pause_ratio": 0.3,
        "pitch_mean": 150,
        "pitch_std": 30,
    }

else:
    # --- Audio mode ---
    status.text(f"Fetching audio for {ticker} {period} {year}...")
    audio_path, audio_result = fetch_audio(ticker, period, year)
    progress.progress(15)

    if audio_path:
        try:
            status.text("Transcribing with Whisper (cached after first run)...")
            transcription = transcribe_audio(audio_path)
            progress.progress(35)
        except Exception as e:
            pipeline_warnings.append(f"Whisper failed: {e}")
            transcription = {"segments": [], "text": ""}

        try:
            status.text("Speaker diarization...")
            mapped_segments = map_speakers(audio_path, transcription)
            # Resolve SPEAKER_XX -> real names using AV transcript if available
            try:
                av_text, _ = fetch_transcript_cached(ticker, period, year)
                if av_text:
                    av_turns, title_map = parse_av_speakers(av_text)
                    mapped_segments = resolve_speaker_names(mapped_segments, av_turns, title_map)
            except Exception:
                pass
            progress.progress(50)
        except Exception as e:
            pipeline_warnings.append(f"Diarization failed: {e}")
            mapped_segments = transcription.get("segments", [])

        try:
            status.text("FinBERT sentiment (batched)...")
            enriched_segments = analyse_segments(mapped_segments)
            progress.progress(60)
        except Exception as e:
            pipeline_warnings.append(f"Sentiment failed: {e}")
            enriched_segments = []

        try:
            status.text("Audio features (cached after first run)...")
            audio_features = extract_audio_features(audio_path)
            progress.progress(75)
        except Exception as e:
            pipeline_warnings.append(f"Audio features failed: {e}")
            audio_features = {}

    else:
        st.warning(f"{audio_result} -- falling back to Alpha Vantage transcript.")
        try:
            transcript_text, err = fetch_transcript_cached(ticker, period, year)
            if not transcript_text:
                st.error(f"Could not retrieve transcript: {err}")
                st.stop()
            st.success("Alpha Vantage transcript retrieved.")
        except Exception as e:
            st.error(f"Transcript fetch failed: {e}")
            st.stop()
        audio_features = {
            "confidence_proxy": 0.5,
            "pause_ratio": 0.3,
            "pitch_mean": 150,
            "pitch_std": 30,
        }
        progress.progress(50)

# Clip coverage
if enriched_segments:
    call_duration_min = max(s.get("end", 0) for s in enriched_segments) / 60
    is_full_call = call_duration_min >= 25
else:
    call_duration_min = 0.0
    is_full_call = False

# Multimodal fusion
status.text("Multimodal analysis...")
try:
    if enriched_segments:
        overall_sentiment = float(np.mean([s["sentiment_score"] for s in enriched_segments]))
        current_positive  = float(np.mean([s.get("positive", 0.0) for s in enriched_segments]))
        current_negative  = float(np.mean([s.get("negative", 0.0) for s in enriched_segments]))
    else:
        _sent = analyse_sentiment(transcript_text or "")
        overall_sentiment = _sent["score"]
        current_positive  = _sent["positive"]
        current_negative  = _sent["negative"]

    multimodal_result = analyse_multimodal(overall_sentiment, audio_features, enriched_segments or None)
except Exception as e:
    pipeline_warnings.append(f"Multimodal fusion failed: {e}")
    multimodal_result = {"mci": 50, "tone_text_divergence": 0.0,
                         "timeline": pd.DataFrame(), "speaker_breakdown": []}
    overall_sentiment = 0.0
    current_positive  = 0.0
    current_negative  = 0.0

progress.progress(88)

# Insights
status.text("Generating insights...")
try:
    hedge_data    = get_hedging_frequency(enriched_segments) if enriched_segments else {"frequency_per_100": 0}
    prepared_segs, qa_segs = split_prepared_vs_qa(enriched_segments)
    prepared_sentiment = float(pd.DataFrame(prepared_segs)["sentiment_score"].mean()) if prepared_segs else None
    qa_sentiment       = float(pd.DataFrame(qa_segs)["sentiment_score"].mean())       if qa_segs       else None
    insights = generate_insights(multimodal_result, hedge_data, prepared_sentiment, qa_sentiment, ticker)
except Exception as e:
    pipeline_warnings.append(f"Insight generation failed: {e}")
    insights = {
        "mci": 50, "mci_label": "N/A", "tone_text_divergence": 0.0, "divergence_label": "N/A",
        "qa_decay": 0.0, "qa_stress": "N/A", "hedge_frequency": 0.0,
        "flags": [], "timeline": pd.DataFrame(), "speaker_breakdown": [], "peer_data": pd.DataFrame(),
    }
    prepared_segs = qa_segs = []

# Talking points, keywords, key insights
try:
    talking_points = extract_talking_points(enriched_segments or None, transcript_text, n=6)
except Exception:
    talking_points = []

try:
    key_insights = extract_key_insights(enriched_segments, n=3)
except Exception:
    key_insights = {"highlights": [], "risk_signals": [], "hedging_moments": []}

try:
    keywords = get_top_keywords(enriched_segments, transcript_text, n=15)
except Exception:
    keywords = []

# Historical: fetch previous 6 quarters (cached -- no API calls on repeat)
prev_quarters = get_previous_quarters(period, year, n=6)
qoq_data = []  # list of {label, sentiment, positive, negative, hedge_freq}

# Current quarter first
current_stats = {
    "label":      f"{period} {year} (current)",
    "sentiment":  overall_sentiment,
    "positive":   current_positive,
    "negative":   current_negative,
    "hedge_freq": insights["hedge_frequency"],
}
qoq_data.append(current_stats)

historical_stats = []
for prev_period, prev_year in prev_quarters:
    try:
        status.text(f"Fetching {ticker} {prev_period} {prev_year} for history...")
        prev_text, _ = fetch_transcript_cached(ticker, prev_period, prev_year)
        if prev_text:
            prev_stats = analyse_transcript_text(prev_text)
            row = {
                "label":      f"{prev_period} {prev_year}",
                "sentiment":  prev_stats["sentiment"],
                "positive":   prev_stats["positive"],
                "negative":   prev_stats["negative"],
                "hedge_freq": prev_stats["hedge_freq"],
            }
            qoq_data.append(row)
            historical_stats.append(prev_stats)
    except Exception:
        pass

# Narrative Shift Index -- sigma vs prior history
nsi = compute_nsi(current_stats, historical_stats)

# Patch peer df with live values for the selected ticker.
# MCI is patched with text_mci (FinBERT-only) so the comparison is fair --
# all peer MCIs are illustrative text-derived values; mixing in audio for
# the selected ticker creates an apples-to-oranges comparison.
try:
    _peer_df = insights["peer_data"]
    _sel_mask = _peer_df["is_selected"]
    _text_mci = multimodal_result.get("text_mci", round(((overall_sentiment + 1) / 2) * 100, 1))
    _peer_df.loc[_sel_mask, "mci"]       = _text_mci
    _peer_df.loc[_sel_mask, "nsi_sigma"] = round(nsi.get("nsi_sigma", 0.0), 2)
    _peer_df.loc[_sel_mask, "qa_stress"] = round(insights.get("qa_decay", 0.0), 2)
    insights["peer_data"] = _peer_df
except Exception:
    pass

progress.progress(100)
status.empty()
progress.empty()

# Event study — runs in parallel with dashboard render (cached after first run)
status.text("Running event study...")
try:
    es_result = run_event_study(ticker, period, year)
except Exception as _es_e:
    es_result = {"error": str(_es_e)}
status.empty()

for w in pipeline_warnings:
    st.warning(w)

# ============================================================
# DASHBOARD HELPERS
# ============================================================

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

# ============================================================
# DASHBOARD
# ============================================================

# --- CSS ---
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; }

/* ---- Global ---- */
.iq-lbl { font-size:0.68rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase;
          color:#4A5568; border-bottom:1px solid #CBD5E0; padding-bottom:3px; margin-bottom:8px; }

/* ---- Header ---- */
.iq-hdr { background:#0D1B2A; border-radius:6px; padding:12px 20px;
          display:flex; align-items:center; gap:16px; margin-bottom:14px; }
.iq-hdr-tk  { font-size:1.5rem; font-weight:700; color:#FFF; letter-spacing:0.04em; }
.iq-hdr-sub { font-size:0.78rem; color:#7A90A8; text-transform:uppercase; letter-spacing:0.07em; }
.iq-hdr-badge { font-size:0.70rem; background:#1E3A5F; color:#7EBAFF;
                padding:2px 9px; border-radius:3px; margin-left:auto; }

/* ---- Takeaways ---- */
.iq-tk { display:flex; align-items:flex-start; gap:10px;
         padding:8px 12px; border-radius:4px; margin-bottom:5px; }
.iq-tk.high   { background:#FDF0EF; border-left:3px solid #C0392B; }
.iq-tk.medium { background:#FEF9EC; border-left:3px solid #C17B00; }
.iq-tk.low    { background:#EEF7F1; border-left:3px solid #1A7F4B; }
.iq-tk-pill { font-size:0.60rem; font-weight:700; text-transform:uppercase;
              padding:1px 6px; border-radius:2px; white-space:nowrap; margin-top:1px; }
.iq-tk.high   .iq-tk-pill { background:#C0392B; color:#FFF; }
.iq-tk.medium .iq-tk-pill { background:#C17B00; color:#FFF; }
.iq-tk.low    .iq-tk-pill { background:#1A7F4B; color:#FFF; }
.iq-tk-msg { font-size:0.82rem; color:#1A2533; font-weight:500; line-height:1.35; }

/* ---- KPI cards ---- */
.iq-kpi { background:#0D1B2A; border-radius:5px; padding:14px 16px; height:100%;
          border-bottom:3px solid #1565C0; }
.iq-kpi.ok    { border-bottom-color:#1A7F4B; }
.iq-kpi.warn  { border-bottom-color:#C17B00; }
.iq-kpi.alert { border-bottom-color:#C0392B; }
.iq-kpi-lbl { font-size:0.65rem; font-weight:600; text-transform:uppercase;
              letter-spacing:0.12em; color:#7A90A8; margin-bottom:4px; }
.iq-kpi-val { font-size:1.55rem; font-weight:700; color:#FFFFFF;
              font-family:'Courier New',monospace; line-height:1.1; }
.iq-kpi-sub { font-size:0.68rem; color:#4A6080; margin-top:3px; }

/* ---- Signal cards ---- */
.iq-sig { border:1px solid #D8DFE8; border-radius:5px; padding:11px 14px;
          margin-bottom:7px; background:#FAFBFC; }
.iq-sig.high   { border-left:4px solid #C0392B; }
.iq-sig.medium { border-left:4px solid #C17B00; }
.iq-sig.low    { border-left:4px solid #1A7F4B; }
.iq-sig-head { display:flex; align-items:center; gap:8px; margin-bottom:4px; }
.iq-sig-title { font-size:0.83rem; font-weight:700; color:#0D1B2A; flex:1; }
.iq-sig-pill { font-size:0.60rem; font-weight:700; text-transform:uppercase;
               padding:1px 7px; border-radius:2px; }
.iq-sig.high   .iq-sig-pill { background:#FDE8E8; color:#C0392B; }
.iq-sig.medium .iq-sig-pill { background:#FEF3CD; color:#C17B00; }
.iq-sig.low    .iq-sig-pill { background:#E8F5EE; color:#1A7F4B; }
.iq-sig-why { font-size:0.76rem; color:#4A5568; margin-bottom:4px; line-height:1.3; }
.iq-sig-meta { font-size:0.68rem; color:#8A9BB0; display:flex; gap:12px; }
.iq-sig-meta span { display:inline-flex; align-items:center; gap:3px; }

/* ---- Talking points ---- */
.iq-tp { display:flex; align-items:flex-start; gap:10px;
         padding:8px 11px; border-bottom:1px solid #E8ECF0; font-size:0.82rem; line-height:1.35; }
.iq-tp:last-child { border-bottom:none; }
.iq-tp-rank { font-size:0.65rem; color:#8A9BB0; font-weight:700; min-width:16px; padding-top:2px; }
.iq-tp-bar  { min-width:48px; padding-top:5px; }
.iq-tp-bg   { background:#E2E8F0; border-radius:2px; height:5px; width:48px; }
.iq-tp-fill { height:5px; border-radius:2px; }
.iq-tp-score { font-size:0.70rem; font-family:'Courier New',monospace; font-weight:700;
               min-width:44px; text-align:right; padding-top:1px; }
.iq-tp-body { flex:1; color:#1A2533; }
.iq-tp-spk  { font-size:0.65rem; color:#6B7D8F; margin-top:2px; }
.iq-tp-time { font-size:0.65rem; color:#8A9BB0; min-width:40px; text-align:right; padding-top:2px; }

/* ---- Insight rows ---- */
.iq-ins { padding:9px 12px; border-radius:4px; margin-bottom:5px; }
.iq-ins.pos { background:#EEF7F1; border-left:3px solid #1A7F4B; }
.iq-ins.neg { background:#FDF0EF; border-left:3px solid #C0392B; }
.iq-ins.hdg { background:#FEF9EC; border-left:3px solid #C17B00; }
.iq-ins-score { font-family:'Courier New',monospace; font-weight:700; font-size:0.72rem; }
.iq-ins-text  { font-size:0.80rem; color:#1A2533; margin-top:3px; line-height:1.3; }
.iq-ins-meta  { font-size:0.66rem; color:#6B7D8F; margin-top:3px; display:flex; gap:10px; }

/* ---- Peer cards ---- */
.iq-peer-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(130px,1fr)); gap:8px; }
.iq-peer { background:#F7F9FC; border:1px solid #D8DFE8; border-radius:5px;
           padding:10px 12px; text-align:center; }
.iq-peer.selected { background:#0D1B2A; border-color:#1565C0; }
.iq-peer-tk   { font-size:0.75rem; font-weight:700; color:#0D1B2A; letter-spacing:0.06em; }
.iq-peer.selected .iq-peer-tk { color:#7EBAFF; }
.iq-peer-mci  { font-size:1.2rem; font-weight:700; font-family:'Courier New',monospace;
                margin:3px 0; }
.iq-peer-sub  { font-size:0.62rem; color:#8A9BB0; }
.iq-peer.selected .iq-peer-sub { color:#4A6080; }
.iq-peer-sig  { font-size:0.60rem; font-weight:700; text-transform:uppercase;
                padding:1px 5px; border-radius:2px; margin-top:4px; display:inline-block; }
.sig-live     { background:#1E3A5F; color:#7EBAFF; }
.sig-positive { background:#E8F5EE; color:#1A7F4B; }
.sig-neutral  { background:#EDF2F7; color:#4A5568; }
.sig-watch    { background:#FEF3CD; color:#C17B00; }

/* ---- Peer table ---- */
.iq-peer-tbl { width:100%; border-collapse:collapse; font-size:0.76rem;
               font-family:'Helvetica Neue',Helvetica,Arial,sans-serif; }
.iq-peer-tbl th { background:#070F17; color:#4A6080; font-weight:600; font-size:0.63rem;
                  text-transform:uppercase; letter-spacing:0.09em; padding:8px 12px;
                  border-bottom:1px solid #1E3A5F; white-space:nowrap; }
.iq-peer-tbl th:first-child { text-align:left; }
.iq-peer-tbl th:not(:first-child) { text-align:right; }
.iq-peer-tbl td { padding:7px 12px; border-bottom:1px solid #0A141E; color:#CBD5E0; white-space:nowrap; }
.iq-peer-tbl td:first-child { text-align:left; }
.iq-peer-tbl td:not(:first-child) { text-align:right; }
.iq-peer-tbl tr.sel-row { background:#0D1B2A; }
.iq-peer-tbl tr:not(.sel-row) { background:#070F17; }
.iq-peer-tbl .pt-rank { color:#2D4A6A; font-size:0.65rem; width:28px; }
.iq-peer-tbl .pt-tk   { font-weight:700; color:#FFFFFF; }
.iq-peer-tbl .pt-live { color:#4A9EE8; font-size:0.58rem; vertical-align:middle;
                         margin-right:4px; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# BUILD RENDER DATA
# ================================================================
mci_val   = insights["mci"]
div_val   = insights["tone_text_divergence"]
hedge_val = insights["hedge_frequency"]
nsi_sigma = nsi.get("nsi_sigma", 0.0)
nsi_n     = nsi.get("n_quarters", 0)
timeline_df = insights["timeline"]

def _cls(v, hi=75, lo=55, inv=False):
    if inv: hi, lo = lo, hi
    if v >= hi: return "ok" if not inv else "alert"
    if v >= lo: return "warn"
    return "alert" if not inv else "ok"

qa_stress_val = insights.get("qa_decay", 0.0)

# Header mode
if transcript_only or not enriched_segments:
    mode_badge, mode_src = "TRANSCRIPT", "Alpha Vantage"
else:
    src = "Cache" if audio_result and "cache/" in audio_result else "YouTube"
    mode_badge = "MULTIMODAL"
    mode_src   = f"{src} + Whisper | {call_duration_min:.0f} min"

# ================================================================
# HEADER
# ================================================================
st.markdown(f"""<div class="iq-hdr">
  <span class="iq-hdr-tk">{ticker}</span>
  <span class="iq-hdr-sub">{period} {year} &nbsp;|&nbsp; Earnings Call Analysis &nbsp;|&nbsp; {mode_src}</span>
  <span class="iq-hdr-badge">{mode_badge}</span>
</div>""", unsafe_allow_html=True)

# ================================================================
# SECTION 1: KEY TAKEAWAYS + KPI BAR
# ================================================================
tk_col, kpi_col = st.columns([1, 2.2])

with tk_col:
    st.markdown('<div class="iq-lbl">Key Takeaways</div>', unsafe_allow_html=True)
    takeaways = _key_takeaways(insights["flags"], nsi, overall_sentiment, hedge_val)
    tk_html = ""
    for sev, msg in takeaways:
        lbl = {"high":"HIGH","medium":"MEDIUM","low":"CLEAR"}[sev]
        tk_html += f"""<div class="iq-tk {sev}">
          <span class="iq-tk-pill">{lbl}</span>
          <span class="iq-tk-msg">{msg}</span>
        </div>"""
    st.markdown(tk_html, unsafe_allow_html=True)

with kpi_col:
    st.markdown('<div class="iq-lbl">Summary Metrics</div>', unsafe_allow_html=True)
    k1,k2,k3,k4,k5 = st.columns(5)
    with k1:
        cls = _cls(mci_val)
        col = "#1A7F4B" if cls=="ok" else "#C17B00" if cls=="warn" else "#C0392B"
        st.markdown(f"""<div class="iq-kpi {cls}">
          <div class="iq-kpi-lbl">MCI</div>
          <div class="iq-kpi-val" style="color:{col}">{mci_val}</div>
          <div class="iq-kpi-sub">{insights['mci_label']}</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        col = "#1A7F4B" if overall_sentiment>=0.05 else "#C0392B" if overall_sentiment<=-0.05 else "#7A90A8"
        st.markdown(f"""<div class="iq-kpi">
          <div class="iq-kpi-lbl">Tone Sentiment</div>
          <div class="iq-kpi-val" style="color:{col}">{overall_sentiment:+.2f}</div>
          <div class="iq-kpi-sub">FinBERT net score</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        dcls = "alert" if div_val<-0.12 else "warn" if div_val<-0.05 else "ok"
        dcol = "#C0392B" if div_val<-0.05 else "#1A7F4B"
        st.markdown(f"""<div class="iq-kpi {dcls}">
          <div class="iq-kpi-lbl">Tone-Text Div.</div>
          <div class="iq-kpi-val" style="color:{dcol}">{div_val:+.2f}</div>
          <div class="iq-kpi-sub">{insights['divergence_label']}</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        ncls = "alert" if nsi_sigma<-1 else "warn" if nsi_sigma<-0.5 else "ok" if nsi_sigma>0.5 else ""
        ncol = "#C0392B" if nsi_sigma<-0.5 else "#1A7F4B" if nsi_sigma>0.5 else "#7A90A8"
        st.markdown(f"""<div class="iq-kpi {ncls}">
          <div class="iq-kpi-lbl">NSI</div>
          <div class="iq-kpi-val" style="color:{ncol}">{nsi_sigma:+.2f}σ</div>
          <div class="iq-kpi-sub">vs {nsi_n}Q history</div>
        </div>""", unsafe_allow_html=True)
    with k5:
        qcls = "alert" if qa_stress_val>0.20 else "warn" if qa_stress_val>0.10 else "ok"
        qcol = "#C0392B" if qa_stress_val>0.20 else "#C17B00" if qa_stress_val>0.10 else "#1A7F4B"
        qa_lbl = "High" if qa_stress_val>0.20 else "Moderate" if qa_stress_val>0.10 else "Low"
        st.markdown(f"""<div class="iq-kpi {qcls}">
          <div class="iq-kpi-lbl">Q&A Stress</div>
          <div class="iq-kpi-val" style="color:{qcol}">{qa_stress_val:+.2f}</div>
          <div class="iq-kpi-sub">{qa_lbl} decay</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Audio features transparency -- only in audio mode
if not transcript_only and enriched_segments:
    _cp   = audio_features.get("confidence_proxy", 0.5)
    _pr   = audio_features.get("pause_ratio", 0.0)
    _rn   = audio_features.get("wav2vec2_raw_norm", None)
    _pm   = audio_features.get("pitch_mean", 0.0)
    _ps   = audio_features.get("pitch_std", 0.0)
    _cv   = round(_ps / _pm, 3) if _pm > 0 else 0.0
    _rn_str = f" (raw norm {_rn:.1f})" if _rn is not None else ""
    st.markdown(
        '<div class="iq-lbl">Audio Signal Sources</div>'
        f'<p style="font-size:0.73rem;color:#6B7D8F;margin:0 0 14px 0;line-height:1.7;">'
        f'<strong style="color:#CBD5E0;">Text Sentiment</strong> &mdash; FinBERT {overall_sentiment:+.3f} &nbsp;&bull;&nbsp; 40% of MCI'
        f'&nbsp;&nbsp;<strong style="color:#CBD5E0;">Vocal Confidence</strong> &mdash; Wav2Vec2 proxy {_cp:.3f}{_rn_str} &nbsp;&bull;&nbsp; 35% of MCI'
        f'&nbsp;&nbsp;<strong style="color:#CBD5E0;">Pause Density</strong> &mdash; pause ratio {_pr:.3f} &nbsp;&bull;&nbsp; 15% of MCI'
        f'&nbsp;&nbsp;<strong style="color:#CBD5E0;">Pitch Stability</strong> &mdash; pitch CV {_cv:.3f} (mean {_pm:.0f} Hz) &nbsp;&bull;&nbsp; 10% of MCI'
        f'</p>',
        unsafe_allow_html=True,
    )

# ================================================================
# SECTION 2: ANALYST SIGNALS + TALKING POINTS
# ================================================================
sig_col, tp_col = st.columns([1, 1.6])

_why_map = {
    "high":   "Indicates a material risk to forward earnings confidence.",
    "medium": "Watch for confirmation in subsequent quarters.",
    "low":    "Supporting signal - broadly consistent with prior guidance.",
}

with sig_col:
    st.markdown('<div class="iq-lbl">Analyst Signals</div>', unsafe_allow_html=True)
    if insights["flags"]:
        for flag in insights["flags"]:
            sev  = flag["severity"]
            attr = _esc(flag.get("attribution", ""))
            msg  = _esc(flag["message"])
            why  = _esc(_why_map.get(sev, ""))
            lbl  = sev.upper()
            spk_tag = ""
            for sp_kw in ["CEO", "CFO", "COO", "Analyst", "Management"]:
                if sp_kw.lower() in flag.get("attribution", "").lower():
                    spk_tag = f'<span>&#128100; {sp_kw}</span>'
                    break
            st.markdown(
                f'<div class="iq-sig {sev}">'
                f'<div class="iq-sig-head"><span class="iq-sig-title">{msg}</span>'
                f'<span class="iq-sig-pill">{lbl}</span></div>'
                f'<div class="iq-sig-why">{why}</div>'
                f'<div class="iq-sig-meta"><span>&#128204; {attr}</span>{spk_tag}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No signals generated.")

with tp_col:
    st.markdown('<div class="iq-lbl">Key Talking Points -- positive to negative</div>', unsafe_allow_html=True)
    if talking_points:
        sorted_tp = sorted(talking_points, key=lambda x: x["sentiment"], reverse=True)
        rows = ""
        for i, tp in enumerate(sorted_tp, 1):
            score    = tp["sentiment"]
            bar_pct  = int(abs(score) * 100)
            bar_col  = "#1A7F4B" if score >= 0 else "#C0392B"
            time_str = f"{tp['time_min']} min" if tp.get("time_min") is not None else ""
            spk = _speaker_for_time(
                enriched_segments, tp.get("time_min") or 0,
                fallback_turns=av_turns, text=tp.get("text", ""),
            )
            spk_html = f'<div class="iq-tp-spk">&#128100; {_esc(spk)}</div>' if spk else ""
            bar_inner = f'<div class="iq-tp-fill" style="width:{bar_pct}%;background:{bar_col}"></div>'
            rows += (
                f'<div class="iq-tp">'
                f'<div class="iq-tp-rank">#{i}</div>'
                f'<div class="iq-tp-bar"><div class="iq-tp-bg">{bar_inner}</div></div>'
                f'<div class="iq-tp-score" style="color:{bar_col}">{score:+.3f}</div>'
                f'<div class="iq-tp-body">{_esc(tp["text"])}{spk_html}</div>'
                f'<div class="iq-tp-time">{time_str}</div>'
                f'</div>'
            )
        st.markdown(
            f'<div style="border:1px solid #D8DFE8;border-radius:5px;background:#FAFBFC;">{rows}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No talking points available.")

st.divider()

# ================================================================
# SECTION 3: SENTIMENT TRAJECTORY
# ================================================================
st.markdown('<div class="iq-lbl">Intra-call Sentiment Trajectory</div>', unsafe_allow_html=True)

if not timeline_df.empty:
    fig = go.Figure()

    # Q&A shading
    if is_full_call:
        qa_min = timeline_df.loc[timeline_df["section"]=="Q&A","time_min"].min()
        if pd.notna(qa_min):
            fig.add_vrect(x0=qa_min, x1=timeline_df["time_min"].max(),
                          fillcolor="rgba(193,123,0,0.07)", line_width=0)
            fig.add_annotation(x=qa_min, y=0.95, xref="x", yref="paper",
                                text="Q&A Start", showarrow=True, arrowhead=2,
                                arrowcolor="#C17B00", font=dict(size=10, color="#C17B00"),
                                ax=30, ay=0)

    fig.add_trace(go.Scatter(
        x=timeline_df["time_min"], y=timeline_df["sentiment_score"],
        name="FinBERT Sentiment", mode="lines",
        line=dict(color="#1565C0", width=2),
    ))

    # Wav2Vec2 flat reference
    if "acoustic_confidence" in timeline_df.columns:
        proxy_val = timeline_df["acoustic_confidence"].iloc[0]
        fig.add_hline(y=proxy_val, line_width=1.5, line_dash="dash", line_color="#C17B00",
                      annotation_text=f"Wav2Vec2 ({proxy_val:+.2f})",
                      annotation_position="top right", annotation_font_size=10,
                      annotation_font_color="#C17B00")

    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#94A3B8")

    _font = "Helvetica Neue, Helvetica, Arial, sans-serif"
    fig.update_layout(
        height=250, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Call time (min)", yaxis_title="Sentiment",
        yaxis_range=[-1, 1],
        plot_bgcolor="#0F1923", paper_bgcolor="#0F1923",
        font=dict(family=_font, size=11, color="#CBD5E0"),
        legend=dict(orientation="h", y=1.1, font=dict(family=_font, size=11, color="#CBD5E0"),
                    bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1E2D3D", zeroline=False,
                     tickfont=dict(family=_font, color="#7A90A8"),
                     title_font=dict(family=_font, color="#7A90A8"))
    fig.update_yaxes(showgrid=True, gridcolor="#1E2D3D", zeroline=False,
                     tickfont=dict(family=_font, color="#7A90A8"),
                     title_font=dict(family=_font, color="#7A90A8"))
    st.plotly_chart(fig, use_container_width=True)
    if not is_full_call:
        st.caption(f"Short clip ({call_duration_min:.0f} min) -- Q&A annotation requires >= 25 min.")
else:
    st.info("Sentiment trajectory requires audio mode with Whisper segments.")

st.divider()

# ================================================================
# SECTION 4: SENTIMENT EXTREMES (with speaker labels)
# ================================================================
st.markdown('<div class="iq-lbl">Sentiment Extremes</div>', unsafe_allow_html=True)
ki1, ki2, ki3 = st.columns(3)

def _ins_spk(segs, time_min, text=""):
    sp = _speaker_for_time(segs, time_min, fallback_turns=av_turns, text=text)
    return f'<span>&#128100; {_esc(sp)}</span>' if sp else ""

with ki1:
    st.markdown("**Management Highlights**")
    if key_insights["highlights"]:
        for h in key_insights["highlights"]:
            spk = _ins_spk(enriched_segments, h["time_min"], text=h["text"])
            st.markdown(
                f'<div class="iq-ins pos">'
                f'<div><span class="iq-ins-score" style="color:#1A7F4B">{h["score"]:+.2f}</span></div>'
                f'<div class="iq-ins-text">{_esc(h["text"])}</div>'
                f'<div class="iq-ins-meta"><span>&#128337; {h["time_min"]} min</span>{spk}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("None detected.")

with ki2:
    st.markdown("**Risk Signals**")
    if key_insights["risk_signals"]:
        for r in key_insights["risk_signals"]:
            spk = _ins_spk(enriched_segments, r["time_min"], text=r["text"])
            st.markdown(
                f'<div class="iq-ins neg">'
                f'<div><span class="iq-ins-score" style="color:#C0392B">{r["score"]:+.2f}</span></div>'
                f'<div class="iq-ins-text">{_esc(r["text"])}</div>'
                f'<div class="iq-ins-meta"><span>&#128337; {r["time_min"]} min</span>{spk}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("None detected.")

with ki3:
    st.markdown("**Hedging Language**")
    if key_insights["hedging_moments"]:
        for h in key_insights["hedging_moments"]:
            spk = _ins_spk(enriched_segments, h["time_min"], text=h["text"])
            words_safe = _esc(", ".join(h["hedge_words"]))
            st.markdown(
                f'<div class="iq-ins hdg">'
                f'<div><span class="iq-ins-score" style="color:#C17B00">{words_safe}</span></div>'
                f'<div class="iq-ins-text">{_esc(h["text"])}</div>'
                f'<div class="iq-ins-meta"><span>&#128337; {h["time_min"]} min</span>{spk}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("None detected.")

st.divider()

# ================================================================
# SECTION 5: HISTORICAL TREND + DIVERGENCE (side by side)
# ================================================================
hist_col, div_col2 = st.columns([1.6, 1])

with hist_col:
    st.markdown('<div class="iq-lbl">Historical Sentiment Trend -- up to 6 quarters</div>', unsafe_allow_html=True)
    if len(qoq_data) > 1:
        qoq_df  = pd.DataFrame(list(reversed(qoq_data)))
        prior   = qoq_data[1]
        current = qoq_data[0]
        d_sent  = round(current["sentiment"] - prior["sentiment"], 3)
        d_hedge = round(current["hedge_freq"] - prior["hedge_freq"], 2)

        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=qoq_df["label"], y=qoq_df["sentiment"],
            mode="lines+markers", name="Net Sentiment",
            line=dict(color="#4A9EE8", width=2), marker=dict(size=7, color="#4A9EE8"),
        ))
        fig_t.add_trace(go.Scatter(
            x=qoq_df["label"], y=qoq_df["hedge_freq"],
            mode="lines+markers", name="Hedging /100w",
            line=dict(color="#E8A83A", width=1.5, dash="dash"), marker=dict(size=5),
            yaxis="y2",
        ))
        fig_t.add_hline(y=0, line_width=1, line_dash="dot", line_color="#2D4A6A")
        _font = "Helvetica Neue, Helvetica, Arial, sans-serif"
        fig_t.update_layout(
            height=230, margin=dict(l=0, r=40, t=10, b=0),
            legend=dict(orientation="h", y=1.1,
                        font=dict(family=_font, size=11, color="#CBD5E0"),
                        bgcolor="rgba(0,0,0,0)"),
            plot_bgcolor="#0F1923", paper_bgcolor="#0F1923",
            font=dict(family=_font, size=11, color="#CBD5E0"),
            yaxis=dict(title="Sentiment", range=[-1, 1], gridcolor="#1E2D3D",
                       tickfont=dict(family=_font, color="#7A90A8"),
                       title_font=dict(family=_font, color="#7A90A8")),
            yaxis2=dict(title="Hedging", overlaying="y", side="right",
                        showgrid=False,
                        tickfont=dict(family=_font, color="#7A90A8"),
                        title_font=dict(family=_font, color="#7A90A8")),
            xaxis=dict(tickfont=dict(family=_font, color="#7A90A8")),
        )
        st.plotly_chart(fig_t, use_container_width=True)

        hc1, hc2 = st.columns(2)
        hc1.metric("Sentiment vs prior Q", f"{current['sentiment']:+.3f}", f"{d_sent:+.3f}")
        hc2.metric("Hedging vs prior Q", f"{current['hedge_freq']:+.2f}", f"{d_hedge:+.2f}")
    else:
        st.info("Fetch prior quarters by running analysis -- transcripts cache locally.")

with div_col2:
    st.markdown('<div class="iq-lbl">Tone-Text Divergence -- Prepared vs Q&A</div>', unsafe_allow_html=True)
    section_div = multimodal_result.get("section_divergence", [])
    if section_div and is_full_call:
        sd_df = pd.DataFrame(section_div)
        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(name="FinBERT", x=sd_df["section"], y=sd_df["sentiment"],
                               marker_color=["#4A9EE8","#E8A83A"], marker_line_width=0))
        fig_d.add_trace(go.Bar(name="Wav2Vec2", x=sd_df["section"], y=sd_df["acoustic"],
                               marker_color=["rgba(74,158,232,0.3)","rgba(232,168,58,0.3)"],
                               marker_line_width=0))
        fig_d.add_hline(y=0, line_width=1, line_dash="dot", line_color="#2D4A6A")
        _font = "Helvetica Neue, Helvetica, Arial, sans-serif"
        fig_d.update_layout(
            barmode="group", height=230, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.1,
                        font=dict(family=_font, size=10, color="#CBD5E0"),
                        bgcolor="rgba(0,0,0,0)"),
            plot_bgcolor="#0F1923", paper_bgcolor="#0F1923",
            font=dict(family=_font, size=11, color="#CBD5E0"),
            yaxis=dict(title="Score", gridcolor="#1E2D3D",
                       tickfont=dict(family=_font, color="#7A90A8"),
                       title_font=dict(family=_font, color="#7A90A8")),
            xaxis=dict(tickfont=dict(family=_font, color="#CBD5E0")),
        )
        fig_d.update_traces(marker_line_width=0)
        st.plotly_chart(fig_d, use_container_width=True)
        st.caption("Gap = divergence. Acoustic below FinBERT = scripted positivity.")
    else:
        st.metric("Overall divergence", f"{insights['tone_text_divergence']:+.2f}",
                  delta=insights["divergence_label"], delta_color="inverse")
        st.caption("Section view requires audio + full call (>= 25 min).")

st.divider()

# ================================================================
# SECTION 6: SPEAKER ATTRIBUTION
# ================================================================
spk_data = [s for s in insights["speaker_breakdown"] if s.get("speaker", "UNKNOWN") not in ("UNKNOWN", "")]
if spk_data and is_full_call:
    st.markdown('<div class="iq-lbl">Speaker Attribution</div>', unsafe_allow_html=True)
    sp_df = pd.DataFrame(spk_data)
    fig_sp = go.Figure()
    for sec, col in [("Prepared","#4A9EE8"),("Q&A","#E8A83A")]:
        sub = sp_df[sp_df["section"]==sec]
        if not sub.empty:
            fig_sp.add_trace(go.Bar(
                name=sec, x=sub["mci"], y=sub["speaker"],
                orientation="h", marker_color=col, marker_line_width=0,
            ))
    fig_sp.update_layout(
        barmode="group", height=max(160, len(sp_df)*30), margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", y=1.1, font_size=11,
                    font_color="#CBD5E0", bgcolor="rgba(0,0,0,0)"),
        plot_bgcolor="#0F1923", paper_bgcolor="#0F1923",
        font=dict(family="Inter", size=11, color="#CBD5E0"),
        xaxis=dict(title="MCI", range=[0,100], gridcolor="#1E2D3D",
                   tickfont=dict(color="#7A90A8")),
        yaxis=dict(tickfont=dict(color="#CBD5E0")),
    )
    st.plotly_chart(fig_sp, use_container_width=True)
    st.divider()

# ================================================================
# SECTION 7: PEER COMPARISON (KPI card grid)
# ================================================================
peer_sector = insights.get("peer_sector", "Peer Group")
st.markdown(f'<div class="iq-lbl">Peer Comparison -- {_esc(peer_sector)}</div>', unsafe_allow_html=True)
st.caption("&#9679; Live values for selected ticker.")

peer_df = insights.get("peer_data", pd.DataFrame())
if not peer_df.empty and "is_selected" in peer_df.columns:
    sig_classes = {"Live": "sig-live", "Positive": "sig-positive",
                   "Neutral": "sig-neutral", "Watch": "sig-watch"}

    def _pc(val, metric):
        """Return colour for a peer table cell value."""
        if metric == "mci":
            return "#1A7F4B" if val >= 75 else "#C17B00" if val >= 55 else "#C0392B"
        if metric == "nsi":
            return "#C0392B" if val > 1.0 else "#C17B00" if val > 0.5 else \
                   "#1A7F4B" if val < -0.5 else "#8A9BB0"
        if metric == "qa":
            return "#C0392B" if val > 0.20 else "#C17B00" if val > 0.10 else "#1A7F4B"
        if metric == "div":
            return "#C0392B" if val < -0.05 else "#1A7F4B" if val > 0.05 else "#8A9BB0"
        return "#8A9BB0"

    thead = (
        '<thead><tr>'
        '<th></th><th>Ticker</th>'
        '<th>MCI</th><th>NSI &sigma;</th><th>Q&amp;A Stress</th>'
        '<th>Divergence</th><th>Signal</th>'
        '</tr></thead>'
    )
    tbody = "<tbody>"
    for _, row in peer_df.iterrows():
        is_sel   = row["is_selected"]
        tr_cls   = "sel-row" if is_sel else ""
        rank     = int(row.get("rank", 0))
        tk       = _esc(row["ticker"])
        live_dot = '<span class="pt-live">&#9679;</span>' if is_sel else ""
        mc       = row["mci"]
        nsi_v    = row.get("nsi_sigma", 0.0)
        qa_v     = row.get("qa_stress", 0.10)
        div_v    = row.get("divergence", 0.0)
        sig      = row.get("signal", "")
        scls     = sig_classes.get(sig, "sig-neutral")
        mci_c    = "#4A9EE8" if is_sel else _pc(mc, "mci")
        tbody += (
            f'<tr class="{tr_cls}">'
            f'<td class="pt-rank">#{rank}</td>'
            f'<td class="pt-tk">{live_dot}{tk}</td>'
            f'<td style="color:{mci_c};font-weight:600">{mc}</td>'
            f'<td style="color:{_pc(nsi_v,"nsi")}">{nsi_v:+.2f}</td>'
            f'<td style="color:{_pc(qa_v,"qa")}">{qa_v:.2f}</td>'
            f'<td style="color:{_pc(div_v,"div")}">{div_v:+.2f}</td>'
            f'<td><span class="iq-peer-sig {scls}">{_esc(sig)}</span></td>'
            f'</tr>'
        )
    tbody += "</tbody>"
    st.markdown(
        f'<div style="overflow-x:auto;border:1px solid #1E3A5F;border-radius:5px;">'
        f'<table class="iq-peer-tbl">{thead}{tbody}</table></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "MCI = FinBERT text sentiment [0–100] for all peers (fair basis). "
        "NSI sigma = narrative shift vs 6Q avg (>1.0 = materially cautious). "
        "Q&A Stress = sentiment decay in analyst questioning phase. Div = tone-text divergence."
    )
else:
    st.info("Peer data unavailable.")

st.divider()

# ================================================================
# SECTION 8: EVENT STUDY — EARNINGS SENSITIVITY ANALYSIS
# ================================================================
st.markdown('<div class="iq-lbl">Event-Study Analysis &mdash; Earnings Sensitivity</div>', unsafe_allow_html=True)
st.caption(
    f"CAPM event study: estimation window t\u221290 to t\u221220 \u2022 event window t\u22121 to t+3 \u2022 market proxy: SPY"
)

_es_col, _sec_col = st.columns([1.1, 1])

with _es_col:
    if "error" in es_result:
        st.warning(f"Event study unavailable: {es_result['error']}")
    else:
        _beta    = es_result["beta"]
        _car     = es_result["car"]
        _tstat   = es_result["t_stat"]
        _rsq     = es_result["r_squared"]
        _n_est   = es_result["n_est"]
        _edate   = es_result["event_date"]
        _ar_df   = es_result["ar_series"]
        _sig     = abs(_tstat) >= 1.96
        _sens_lbl = "High" if _beta > 0.7 else "Medium" if _beta > 0.3 else "Low"
        _beta_col = "#C0392B" if _beta > 0.7 else "#C17B00" if _beta > 0.3 else "#1A7F4B"
        _car_col  = "#1A7F4B" if _car >= 0 else "#C0392B"
        _sig_str  = "Significant (95%)" if _sig else "Not significant"
        _sig_col  = "#1A7F4B" if _sig else "#7A90A8"

        # KPI cards
        _ek1, _ek2, _ek3, _ek4 = st.columns(4)
        _ek1.markdown(f"""<div class="iq-kpi ok">
          <div class="iq-kpi-lbl">Beta (&beta;)</div>
          <div class="iq-kpi-val" style="color:{_beta_col}">{_beta:.2f}</div>
          <div class="iq-kpi-sub">{_sens_lbl} sensitivity</div>
        </div>""", unsafe_allow_html=True)
        _ek2.markdown(f"""<div class="iq-kpi {'ok' if _car>=0 else 'alert'}">
          <div class="iq-kpi-lbl">CAR [-1,+3]</div>
          <div class="iq-kpi-val" style="color:{_car_col}">{_car:+.2%}</div>
          <div class="iq-kpi-sub">Cumul. abnormal ret.</div>
        </div>""", unsafe_allow_html=True)
        _ek3.markdown(f"""<div class="iq-kpi ok">
          <div class="iq-kpi-lbl">T-Statistic</div>
          <div class="iq-kpi-val" style="color:{_sig_col}">{_tstat:+.2f}</div>
          <div class="iq-kpi-sub">{_sig_str}</div>
        </div>""", unsafe_allow_html=True)
        _ek4.markdown(f"""<div class="iq-kpi ok">
          <div class="iq-kpi-lbl">Model R&sup2;</div>
          <div class="iq-kpi-val" style="color:#CBD5E0">{_rsq:.2f}</div>
          <div class="iq-kpi-sub">{_n_est} est. days &bull; t=0: {_edate}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # AR bar chart + CAR line
        _font = "Helvetica Neue, Helvetica, Arial, sans-serif"
        fig_es = go.Figure()

        _bar_colours = ["#1A7F4B" if v >= 0 else "#C0392B" for v in _ar_df["ar"]]
        fig_es.add_trace(go.Bar(
            x=_ar_df["label"],
            y=_ar_df["ar"],
            name="Abnormal Return",
            marker_color=_bar_colours,
            marker_line_width=0,
            yaxis="y1",
        ))
        fig_es.add_trace(go.Scatter(
            x=_ar_df["label"],
            y=_ar_df["car_cumulative"],
            name="Cumulative AR",
            mode="lines+markers",
            line=dict(color="#4A9EE8", width=2),
            marker=dict(size=7, color="#4A9EE8"),
            yaxis="y1",
        ))

        # Vertical line at t=0
        _t0_label = f"t=0"
        if _t0_label in _ar_df["label"].values:
            fig_es.add_vline(
                x=list(_ar_df["label"]).index(_t0_label),
                line_dash="dot",
                line_color="#7A90A8",
                annotation_text="Event (t=0)",
                annotation_font_color="#7A90A8",
                annotation_font_size=10,
            )

        fig_es.update_layout(
            height=260,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="#0F1923",
            paper_bgcolor="#0F1923",
            font=dict(family=_font, size=11, color="#CBD5E0"),
            legend=dict(orientation="h", y=1.08, font=dict(family=_font, size=10, color="#CBD5E0"), bgcolor="rgba(0,0,0,0)"),
            yaxis=dict(
                title="Return", gridcolor="#1E2D3D", zeroline=True, zerolinecolor="#2A4060",
                tickformat=".1%", tickfont=dict(family=_font, color="#7A90A8"),
                title_font=dict(family=_font, color="#7A90A8"),
            ),
            xaxis=dict(tickfont=dict(family=_font, color="#7A90A8")),
            barmode="overlay",
        )
        st.plotly_chart(fig_es, use_container_width=True)
        st.caption(
            f"Bars = daily abnormal return (actual \u2212 CAPM expected). "
            f"Blue line = cumulative CAR. &beta; = {_beta:.3f} from {_n_est}-day estimation window."
        )

with _sec_col:
    st.markdown('<div style="color:#7A90A8;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;">Sector Sensitivity Index (\u03b2)</div>', unsafe_allow_html=True)

    _live_beta  = es_result.get("beta") if "error" not in es_result else None
    _peer_sector_label = insights.get("peer_sector", None)
    _sec_df = get_sector_sensitivity_df(_peer_sector_label, _live_beta)

    _bar_cols = [
        "#4A9EE8" if row["is_live"] else row["colour"]
        for _, row in _sec_df.iterrows()
    ]
    _font = "Helvetica Neue, Helvetica, Arial, sans-serif"
    fig_sec = go.Figure()
    fig_sec.add_trace(go.Bar(
        x=_sec_df["beta"],
        y=_sec_df["sector"],
        orientation="h",
        marker_color=_bar_cols,
        marker_line_width=0,
        text=[f"{b:.2f}" for b in _sec_df["beta"]],
        textposition="outside",
        textfont=dict(family=_font, size=10, color="#CBD5E0"),
        hovertemplate="<b>%{y}</b><br>β = %{x:.3f}<extra></extra>",
    ))

    # Sensitivity threshold lines
    for _thresh, _lbl in [(0.3, "Low | Med"), (0.7, "Med | High")]:
        fig_sec.add_vline(
            x=_thresh,
            line_dash="dot",
            line_color="#2A4060",
            annotation_text=_lbl,
            annotation_font_color="#7A90A8",
            annotation_font_size=9,
            annotation_position="top",
        )

    fig_sec.update_layout(
        height=520,
        margin=dict(l=0, r=40, t=10, b=0),
        plot_bgcolor="#0F1923",
        paper_bgcolor="#0F1923",
        font=dict(family=_font, size=10, color="#CBD5E0"),
        showlegend=False,
        xaxis=dict(
            title="Beta (β)", range=[0, max(_sec_df["beta"]) * 1.25],
            gridcolor="#1E2D3D", tickfont=dict(family=_font, color="#7A90A8"),
            title_font=dict(family=_font, color="#7A90A8"),
        ),
        yaxis=dict(tickfont=dict(family=_font, color="#CBD5E0")),
    )
    st.plotly_chart(fig_sec, use_container_width=True)
    st.caption(
        "\U0001f7e2 Low (<0.3) \u2003\U0001f7e1 Medium (0.3\u20130.7) \u2003\U0001f534 High (>0.7) \u2003"
        "\U0001f535 Selected sector (live \u03b2)"
    )
