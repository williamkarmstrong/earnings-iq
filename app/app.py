"""
Streamlit app for multimodal earnings analysis.
This is the main entry point for the app and 
builds the dashboard that the user will see.

This file connects all parts of the system together and runs the full pipeline:
1. Fetch earnings call data 
    - Cache or yt dlp for audio files
    - Alpha Vantage for backup transcripts or transcript only mode
2. Process audio/transcript 
    - Whisper for transcription
    - pyannote for speaker diarization
3. Analyse sentiment and tone 
    - FinBERT for sentiment
    - Wav2Vec2 for confidence proxy
    - Librosa for acoustics
4. Generate insights & Multimodal Analysis
5. Display results in the dashboard (Streamlit)
"""
import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ingestion import fetch_audio, fetch_transcript
from event_study import run_event_study, get_sector_sensitivity_df, compute_sector_earnings_sensitivity
from speech import transcribe_audio, map_speakers, resolve_speaker_names, is_management_speaker
from nlp import (
    analyse_segments, analyse_sentiment, get_hedging_frequency,
    split_prepared_vs_qa, get_top_keywords, extract_key_insights,
    extract_talking_points, analyse_transcript_text, compute_nsi,
    parse_av_speakers,
    weighted_segment_mean, find_qa_start_time
)
from audio import extract_audio_features
from multimodal import analyse_multimodal
from insights import generate_insights, SIGNAL_MCI_POSITIVE, SIGNAL_MCI_WATCH
from utils import _analyse_peer, get_previous_quarters, _speaker_from_turns, _speaker_for_time, _key_takeaways, is_valid_ticker

def find_audio_file(ticker: str, period: str, year: int) -> str | None:
    """Return absolute path to a cached audio file for this call, or None."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = os.path.join(project_root, "cache", f"{ticker}_{year}_{period}")
    for ext in ("wav", "webm", "mp3"):
        path = f"{base}.{ext}"
        if os.path.isfile(path):
            return path
    return None


st.set_page_config(page_title="EarningsIQ", layout="wide")

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
    year = st.slider("Year", 2011, 2026, 2018)

    transcript_only = st.toggle(
        "Transcript-only mode",
        value=False,
        help="ON: skips audio (Whisper/Wav2Vec2/librosa) uses Alpha Vantage transcript only. "
             "Faster for UI testing. OFF (default): full multimodal pipeline.",
    )
    if transcript_only:
        st.caption("Mode: Transcript only")
    else:
        st.caption("Mode: Audio + Text (default)")

    if st.button("Analyse Call", use_container_width=True, type="primary"):
        st.session_state.analysis_run = True
        st.session_state.analysis_key = (ticker, period, year, transcript_only)
        st.session_state.pop("audio_autoplay",  None)
        st.session_state.pop("audio_seek_time", None)
    elif st.session_state.get("analysis_key") != (ticker, period, year, transcript_only):
        st.session_state.pop("analysis_run", None)
    st.divider()

    with st.expander("Metrics guide"):
        st.markdown("""
**MCI (0-100):** FinBERT 40% + Wav2Vec2 35% + Pause 15% + Pitch 10% *(full multimodal)*
Low ≤52 · Medium 52–56 · High ≥56

**MCI in peer table:** FinBERT text only for fair cross peer comparison

**NSI (sigma):** Sentiment z-score vs prior 6 quarters

**Divergence:** Acoustic vs textual gap scaled to ±0.5 (negative = scripted positivity; flag at −0.12)

**Hedging /100w:** Uncertainty word density (flag threshold: 8.0)

**Q&A Stress:** Sentiment drop from prepared remarks to Q&A (65% split)
        """)

# ============================================================
# MAIN
# ============================================================
st.title("EarningsIQ - Multimodal Earnings Call Analysis")

if not st.session_state.get("analysis_run", False):
    st.info("Select a ticker and quarter in the sidebar, then click **Analyse Call**.")
    st.stop()

if not is_valid_ticker(ticker):
    st.error(f"'{ticker}' is not a recognised ticker.")
    st.stop()

# ============================================================
# PIPELINE
# ============================================================
MIN_FULL_CALL_MINUTES = 25

@st.cache_data(show_spinner=False)
def run_pipeline(ticker: str, period: str, year: int, transcript_only: bool) -> dict:
    """
    Run the full earnings call analysis pipeline.
    Returns a result dict consumed by the dashboard.
    """
    progress = st.progress(0)
    status   = st.empty()
    pipeline_warnings = []

    enriched_segments = []
    transcript_text   = None
    audio_features    = {}
    av_turns          = []
    title_map         = {}
    audio_result      = ""
    is_demo_cached    = False
    demo_mode         = False
    demo_json         = ""
    qoq_data_cached   = None
    nsi_cached        = None
    peer_data_cached  = None
    es_result_cached  = None

    # 1. Fetch Transcript from Alpha Vantage for Backup and Speaker Mapping
    # Skip in audio mode when a demo cache exists — av_turns/title_map come from cache instead.
    _demo_dir = f"demo/{ticker}_{year}_{period}"
    _demo_cache_exists = not transcript_only and os.path.isdir(_demo_dir) and os.path.exists(f"{_demo_dir}/results.json")
    if not _demo_cache_exists:
        av_json, av_err = fetch_transcript(ticker, period, year)
        if av_json:
            av_turns, title_map = parse_av_speakers(av_json)
        else:
            av_turns = []
            title_map = {}
    else:
        av_json, av_err = None, None

    # 2. Transcript Only Mode: Check Transcript Exists
    if transcript_only:
        if not av_json:
            st.error(f"Could not load transcript: {av_err}")
            st.stop()

        progress.progress(50)
        st.success("Transcript loaded (transcript only mode, audio analysis skipped).")

        audio_features = {
            "confidence_proxy": 0.5,
            "pause_ratio": 0.3,
            "pitch_mean": 150,
            "pitch_std": 30,
        }
    else:
        # 3. Demo Mode: Try Load Demo Results and Stop
        demo_dir = _demo_dir
        demo_json = f"{demo_dir}/results.json"
        demo_mode = os.path.isdir(demo_dir)
        is_demo_cached = _demo_cache_exists

        if is_demo_cached:
            status.text("Loading fully processed results from demo cache...")
            try:
                with open(demo_json, "r", encoding="utf-8") as f:
                    data = json.load(f)

                enriched_segments = data.get("enriched_segments", [])
                audio_features    = data.get("audio_features", {})
                transcript_text   = data.get("transcript_text", "")
                av_turns          = data.get("av_turns", [])
                title_map         = data.get("title_map", {})
                qoq_data_cached   = data.get("qoq_data", [])
                nsi_cached        = data.get("nsi", {})
                peer_data_cached  = data.get("peer_data", [])
                es_result_cached  = data.get("es_result", {})

                audio_result = f"{demo_dir} cache"
                audio_path   = f"{demo_dir}/audio.mp3"
                progress.progress(80)
            except Exception as e:
                st.error(f"Failed to load demo cache: {e}")
                st.stop()

        else:
            # 4. Audio Mode: Fetch Audio and Run Specifics
            status.text(f"Fetching audio for {ticker} {period} {year}...")
            audio_path, audio_result = fetch_audio(ticker, period, year)
            progress.progress(15)

            if audio_path:
                # A. TRANSCRIPTION
                status.text("Transcribing with Whisper...")
                transcription = transcribe_audio(audio_path)
                progress.progress(35)

                # B. DIARIZATION & NAME RESOLUTION
                status.text("Speaker diarization...")
                mapped_segments   = map_speakers(audio_path, transcription)
                enriched_segments = resolve_speaker_names(mapped_segments, av_turns, title_map)
                progress.progress(50)

                # C. SENTIMENT & FEATURES
                status.text("FinBERT sentiment analysis...")
                enriched_segments = analyse_segments(enriched_segments)
                progress.progress(60)
                status.text("Extracting audio features...")
                audio_features = extract_audio_features(audio_path)
                progress.progress(75)
            else:
                st.warning(f"Failed to fetch audio for {ticker} {period} {year}.")
                status.text("Defaulting to transcript only mode.")

                if not av_json:
                    st.error(f"Could not retrieve transcript: {av_err}")
                    st.stop()

                transcript_only = True
                audio_features = {
                    "confidence_proxy": 0.5,
                    "pause_ratio": 0.3,
                    "pitch_mean": 150,
                    "pitch_std": 30,
                }
                progress.progress(50)

    # 5. Continue Running Rest of Pipeline
    if enriched_segments:
        call_duration_min = max(s.get("end", 0) for s in enriched_segments) / 60
        is_full_call = call_duration_min >= MIN_FULL_CALL_MINUTES
    else:
        call_duration_min = 0.0
        is_full_call = False

    # Filter to management-only segments for MCI and Q&A analysis.
    if enriched_segments:
        _mgmt = [s for s in enriched_segments if is_management_speaker(s.get("speaker", "")) is not False]
        management_segments = _mgmt if _mgmt else enriched_segments
    else:
        management_segments = []

    # Detect Q&A start from all segments
    qa_start_time = find_qa_start_time(enriched_segments) if enriched_segments else None

    # Multimodal fusion
    status.text("Multimodal analysis...")
    try:
        if management_segments:
            overall_sentiment = weighted_segment_mean(management_segments, "sentiment_score")
            current_positive  = weighted_segment_mean(management_segments, "positive")
            current_negative  = weighted_segment_mean(management_segments, "negative")
        else:
            _sent = analyse_sentiment(transcript_text or "")
            overall_sentiment = _sent["score"]
            current_positive  = _sent["positive"]
            current_negative  = _sent["negative"]

        multimodal_result = analyse_multimodal(
            overall_sentiment, audio_features,
            management_segments or None,
            qa_start_time=qa_start_time,
        )
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
        hedge_data = get_hedging_frequency(management_segments) if management_segments else {"frequency_per_100": 0}
        # Pass all enriched_segments as search source so operator's Q&A announcement is found
        prepared_segs, qa_segs = split_prepared_vs_qa(management_segments, search_segments=enriched_segments)
        prepared_sentiment = weighted_segment_mean(prepared_segs, "sentiment_score") if prepared_segs else None
        qa_sentiment       = weighted_segment_mean(qa_segs,       "sentiment_score") if qa_segs       else None
        insights = generate_insights(multimodal_result, hedge_data, prepared_sentiment, qa_sentiment, ticker)
    except Exception as e:
        pipeline_warnings.append(f"Insight generation failed: {e}")
        insights = {
            "mci": 50, "mci_label": "N/A", "tone_text_divergence": 0.0, "divergence_label": "N/A",
            "qa_decay": 0.0, "qa_stress": "N/A", "hedge_frequency": 0.0,
            "flags": [], "timeline": pd.DataFrame(), "speaker_breakdown": [], "peer_data": pd.DataFrame(), "peer_tickers": [], "peer_sector": "Peer Group",
        }
        prepared_segs = qa_segs = []

    # Talking points, keywords, key insights
    try:
        talking_points = extract_talking_points(management_segments or None, transcript_text, n=6)
    except Exception:
        talking_points = []

    try:
        key_insights = extract_key_insights(management_segments, n=5)
    except Exception:
        key_insights = {"highlights": [], "risk_signals": []}

    try:
        keywords = get_top_keywords(management_segments, transcript_text, n=15)
    except Exception:
        keywords = []

    # If offline demo data is cached, bypass all remaining API calls
    if is_demo_cached:
        qoq_data = qoq_data_cached
        nsi      = nsi_cached
        insights["peer_data"] = pd.DataFrame(peer_data_cached)
        es_result = es_result_cached
        if isinstance(es_result.get("ar_series"), list):
            es_result["ar_series"] = pd.DataFrame(es_result["ar_series"])
        status.empty()
        progress.empty()
    else:
        # Historical: fetch previous 4 quarters (cached -- no API calls on repeat)
        prev_quarters = get_previous_quarters(period, year, n=4)
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
                prev_json, _ = fetch_transcript(ticker, prev_period, prev_year)
                if prev_json:
                    prev_text  = " ".join(item.get("content", "") for item in prev_json)
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
            except Exception as e:
                pipeline_warnings.append(f"Could not fetch {ticker} {prev_period} {prev_year} for QoQ history: {e}")

        # Narrative Shift Index -- sigma vs prior history
        nsi = compute_nsi(current_stats, historical_stats)

        # Build real peer comparison DataFrame by fetching/analysing peer transcripts.
        # Uses st.cache_data so previously-analysed tickers load instantly.
        _peer_tickers  = insights.get("peer_tickers", [])
        _live_text_mci = multimodal_result.get("text_mci", round(((overall_sentiment + 1) / 2) * 100, 1))
        _live_qa_stress = round(insights.get("qa_decay", 0.0), 3)
        _live_signal = "Positive" if _live_text_mci >= SIGNAL_MCI_POSITIVE else "Watch" if _live_text_mci <= SIGNAL_MCI_WATCH else "Neutral"

        _peer_rows = []
        for _pt in _peer_tickers:
            if _pt.upper() == ticker.upper():
                _peer_rows.append({
                    "ticker":      ticker.upper(),
                    "mci":         _live_text_mci,
                    "qa_stress":   _live_qa_stress,
                    "signal":      _live_signal,
                    "is_selected": True,
                })
            else:
                _res = _analyse_peer(_pt, period, year)
                if _res:
                    _peer_rows.append({
                        "ticker":      _pt.upper(),
                        "mci":         _res["mci"],
                        "qa_stress":   _res["qa_stress"],
                        "signal":      _res["signal"],
                        "is_selected": False,
                    })
                else:
                    _peer_rows.append({
                        "ticker":      _pt.upper(),
                        "mci":         None,
                        "qa_stress":   None,
                        "signal":      "N/A",
                        "is_selected": False,
                    })

        if _peer_rows:
            _peer_df = pd.DataFrame(_peer_rows)
            _peer_df = _peer_df.sort_values(
                "mci",
                ascending=False,
                key=lambda s: s.fillna(-1)
            ).reset_index(drop=True)
            _peer_df["rank"] = _peer_df.index + 1
            _valid_mci = _peer_df.loc[~_peer_df["is_selected"] & _peer_df["mci"].notna(), "mci"]
            _peer_avg  = _valid_mci.mean() if not _valid_mci.empty else _live_text_mci
            _peer_df["delta_vs_peers"] = (_peer_df["mci"] - _peer_avg).round(1)
            insights["peer_data"] = _peer_df
        else:
            insights["peer_data"] = pd.DataFrame()

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

        # SAVE ALL NATIVE AND OFFLINE CACHING
        if demo_mode and enriched_segments:
            try:
                _es_serial = es_result.copy()
                if "ar_series" in _es_serial and isinstance(_es_serial["ar_series"], pd.DataFrame):
                    _es_serial["ar_series"] = _es_serial["ar_series"].to_dict("records")

                with open(demo_json, "w", encoding="utf-8") as f:
                    json.dump({
                        "enriched_segments": enriched_segments,
                        "audio_features":    audio_features,
                        "transcript_text":   transcript_text if transcript_text is not None else "",
                        "av_turns":          av_turns,
                        "title_map":         title_map,
                        "qoq_data":          qoq_data,
                        "nsi":               nsi,
                        "peer_data":         insights["peer_data"].to_dict("records") if not insights["peer_data"].empty else [],
                        "es_result":         _es_serial,
                    }, f, default=str)
            except Exception as e:
                pipeline_warnings.append(f"Failed to save demo cache: {e}")

    for w in pipeline_warnings:
        st.warning(w)

    return {
        "enriched_segments":  enriched_segments,
        "management_segments": management_segments,
        "overall_sentiment":  overall_sentiment,
        "multimodal_result":  multimodal_result,
        "audio_features":     audio_features,
        "insights":           insights,
        "qoq_data":           qoq_data,
        "nsi":                nsi,
        "es_result":          es_result,
        "talking_points":     talking_points,
        "key_insights":       key_insights,
        "keywords":           keywords,
        "av_turns":           av_turns,
        "qa_start_time":      qa_start_time,
        "call_duration_min":  call_duration_min,
        "is_full_call":       is_full_call,
        "transcript_only":    transcript_only,
        "audio_result":       audio_result,
    }


result = run_pipeline(ticker, period, year, transcript_only)

enriched_segments   = result["enriched_segments"]
management_segments = result["management_segments"]
overall_sentiment   = result["overall_sentiment"]
multimodal_result   = result["multimodal_result"]
audio_features      = result["audio_features"]
insights            = result["insights"]
qoq_data            = result["qoq_data"]
nsi                 = result["nsi"]
es_result           = result["es_result"]
talking_points      = result["talking_points"]
key_insights        = result["key_insights"]
keywords            = result["keywords"]
av_turns            = result["av_turns"]
qa_start_time       = result["qa_start_time"]
call_duration_min   = result["call_duration_min"]
is_full_call        = result["is_full_call"]
transcript_only     = result["transcript_only"]
audio_result        = result["audio_result"]

audio_file_path = None if (transcript_only or not enriched_segments) else find_audio_file(ticker, period, year)

# ============================================================
# DASHBOARD
# ============================================================
mci_val       = insights["mci"]
div_val       = insights["tone_text_divergence"]
hedge_val     = insights["hedge_frequency"]
nsi_sigma     = nsi.get("nsi_sigma", 0.0)   
nsi_n         = nsi.get("n_quarters", 0)
timeline_df   = insights["timeline"]
qa_stress_val = insights.get("qa_decay", 0.0)

if transcript_only or not enriched_segments:
    mode_badge, mode_src = "TRANSCRIPT", "Alpha Vantage"
else:
    src        = "EarningsCallBiz" if audio_result and "cache" in audio_result else "YouTube"
    mode_badge = "MULTIMODAL"
    mode_src   = f"{src} + Whisper | {call_duration_min:.0f} min"

# ================================================================
# HEADER
# ================================================================
st.title(f"{ticker}  ·  {period} {year} Earnings Call Analysis")
st.caption(f"{mode_badge}  ·  {mode_src}")
st.divider()

# ================================================================
# SECTION 1: KEY TAKEAWAYS + KPI BAR
# ================================================================
tk_col, kpi_col = st.columns([1, 2.2])

with tk_col:
    st.subheader("Key Takeaways")
    takeaways = _key_takeaways(insights["flags"], nsi, overall_sentiment, hedge_val)
    for sev, msg in takeaways:
        if sev == "high":
            st.error(msg)
        elif sev == "medium":
            st.warning(msg)
        else:
            st.success(msg)

with kpi_col:
    st.subheader("Summary Metrics")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("MCI", mci_val, help="Management Confidence Index [0–100]")
    k2.metric("Transcript Sentiment", f"{overall_sentiment:+.2f}", help="FinBERT net score")
    if transcript_only or not audio_features.get("wav2vec2_raw_norm"):
        k3.metric("Tone-Text Div.", "N/A", help="Audio required")
    else:
        k3.metric("Tone-Text Div.", f"{div_val:+.2f}", help=insights["divergence_label"])
    k4.metric("NSI", f"{nsi_sigma:+.2f}σ", help=f"Narrative Shift Index vs {nsi_n}Q history")
    k5.metric("Q&A Stress", f"{qa_stress_val:+.2f}", help="Sentiment decay from prepared remarks to Q&A")

if not transcript_only and enriched_segments:
    _cp  = audio_features.get("confidence_proxy", 0.5)
    _pr  = audio_features.get("pause_ratio", 0.0)
    _rn  = audio_features.get("wav2vec2_raw_norm")
    _pm  = audio_features.get("pitch_mean", 0.0)
    _ps  = audio_features.get("pitch_std", 0.0)
    _cv  = round(_ps / _pm, 3) if _pm > 0 else 0.0
    with st.expander("Audio Signal Sources"):
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Text Sentiment", f"{overall_sentiment:+.3f}", help="FinBERT net score · 40% of MCI")
        _rn_help = f", raw norm {_rn:.1f}" if _rn else ""
        a2.metric("Vocal Confidence", f"{_cp:.3f}", help=f"Wav2Vec2 proxy{_rn_help} · 35% of MCI")
        a3.metric("Pause Density", f"{_pr:.3f}", help="Pause ratio · 15% of MCI")
        a4.metric("Pitch Stability", f"{_cv:.3f}", help=f"Pitch CV (mean {_pm:.0f} Hz) · 10% of MCI")

st.divider()

# ================================================================
# SECTION 2: ANALYST SIGNALS + TALKING POINTS
# ================================================================
sig_col, tp_col = st.columns([1, 1.6])

_has_audio = not transcript_only and bool(audio_features.get("wav2vec2_raw_norm"))
_visible_flags = [
    f for f in insights["flags"]
    if _has_audio or "divergence" not in f.get("message", "").lower()
]

with sig_col:
    st.subheader("Analyst Signals")
    if _visible_flags:
        for flag in _visible_flags:
            sev  = flag["severity"]
            msg  = flag["message"]
            attr = flag.get("attribution", "")
            body = f"**{msg}**\n\n{attr}" if attr else f"**{msg}**"
            if sev == "high":
                st.error(body)
            elif sev == "medium":
                st.warning(body)
            else:
                st.success(body)
    else:
        extra = "" if _has_audio else " Enable audio mode for divergence signals."
        st.caption(f"No signals generated.{extra}")

with tp_col:
    st.subheader("Key Talking Points")
    if talking_points:
        for i, tp in enumerate(talking_points, 1):
            score    = tp["sentiment"]
            time_str = f"{tp['time_min']} min" if tp.get("time_min") is not None else ""
            spk = _speaker_for_time(
                management_segments, tp.get("time_min") or 0,
                fallback_turns=av_turns, text=tp.get("text", ""),
            )
            meta = " · ".join(x for x in [time_str, spk] if x)
            _can_play = audio_file_path and tp.get("time_min") is not None
            if _can_play:
                _b, _t = st.columns([1, 9])
                with _b:
                    if st.button("▶", key=f"tp_play_{i}", help=f"Play from {tp['time_min']} min", use_container_width=True):
                        st.session_state.audio_seek_time = int(float(tp["time_min"]) * 60)
                        st.session_state.audio_autoplay  = True
                        st.rerun()
                with _t:
                    st.markdown(f"**#{i}** `{score:+.3f}` {tp['text']}")
                    if meta:
                        st.caption(meta)
            else:
                st.markdown(f"**#{i}** `{score:+.3f}` {tp['text']}")
                if meta:
                    st.caption(meta)
    else:
        st.caption("No talking points available.")

st.divider()

# ================================================================
# SECTION 3: SENTIMENT TRAJECTORY
# ================================================================
st.subheader("Intra-call Sentiment Trajectory")

if not timeline_df.empty:
    fig = go.Figure()
    if is_full_call:
        qa_min = timeline_df.loc[timeline_df["section"] == "Q&A", "time_min"].min()
        if pd.notna(qa_min):
            _qa_method = "detected" if qa_start_time is not None else "65% heuristic"
            fig.add_vrect(x0=qa_min, x1=timeline_df["time_min"].max(),
                          fillcolor="rgba(255,165,0,0.08)", line_width=0)
            fig.add_annotation(x=qa_min, y=0.95, xref="x", yref="paper",
                                text=f"Q&A Start ({_qa_method})",
                                showarrow=True, arrowhead=2, ax=30, ay=0)
    fig.add_trace(go.Scatter(
        x=timeline_df["time_min"], y=timeline_df["sentiment_score"],
        name="FinBERT Sentiment", mode="lines+markers",
        line=dict(width=2), marker=dict(size=5, opacity=0.7),
    ))
    if "acoustic_confidence" in timeline_df.columns:
        proxy_val = timeline_df["acoustic_confidence"].mean()
        fig.add_hline(y=proxy_val, line_width=1.5, line_dash="dash",
                      annotation_text=f"Wav2Vec2 ({proxy_val:+.2f})",
                      annotation_position="top right")
    fig.add_hline(y=0, line_width=1, line_dash="dot")
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="Call time (min)", yaxis_title="Sentiment",
                      yaxis_range=[-1, 1])
    _chart_evt = st.plotly_chart(
        fig, use_container_width=True,
        on_select="rerun", selection_mode="points", key="sentiment_traj",
    )
    if audio_file_path and _chart_evt and _chart_evt.selection and _chart_evt.selection.points:
        _pt = _chart_evt.selection.points[0]
        _t  = _pt.get("x") if isinstance(_pt, dict) else getattr(_pt, "x", None)
        if _t is not None:
            st.session_state.audio_seek_time = int(float(_t) * 60)
            st.session_state.audio_autoplay  = True

    if not is_full_call:
        st.caption(f"Short clip ({call_duration_min:.0f} min) — Q&A annotation requires ≥ 25 min.")
else:
    st.info("Sentiment trajectory requires audio mode with Whisper segments.")

st.divider()

# Audio player — placed here so chart event (above) and talking-point reruns
# both land here with the correct state. Sentiment extremes buttons (below)
# overwrite the slot directly in the same run without needing a rerun.
_autoplay = st.session_state.get("audio_autoplay", False)
st.session_state.audio_autoplay = False  # always consume
audio_slot    = None
_audio_fmt    = None
if audio_file_path:
    _seek  = st.session_state.get("audio_seek_time", 0)
    _ext   = os.path.splitext(audio_file_path)[1].lstrip(".")
    _audio_fmt = {"wav": "audio/wav", "webm": "audio/webm", "mp3": "audio/mpeg"}.get(_ext, "audio/wav")
    _mm, _ss   = divmod(int(_seek), 60)
    _cue_lbl   = f"Cued to {_mm}:{_ss:02d}" if _seek > 0 else "Click a chart point or ▶ button to seek"
    st.caption(f"Earnings call audio · {_cue_lbl}")
    audio_slot = st.empty()
    audio_slot.audio(audio_file_path, format=_audio_fmt, start_time=_seek, autoplay=_autoplay)

st.divider()

# ================================================================
# SECTION 4: SENTIMENT EXTREMES
# ================================================================
st.subheader("Sentiment Extremes")
ki1, ki2 = st.columns(2)

with ki1:
    st.markdown("**Management Highlights**")
    if key_insights["highlights"]:
        for _hi, h in enumerate(key_insights["highlights"]):
            _raw = h.get("speaker", "")
            spk = _raw if _raw not in ("", "UNKNOWN") else _speaker_for_time(
                management_segments, h["time_min"], fallback_turns=av_turns, text=h["text"])
            st.success(h["text"])
            meta = " · ".join(x for x in [spk, f'{h["score"]:+.2f}', f'{h["time_min"]} min'] if x)
            _mc, _bc = st.columns([8, 1])
            with _mc:
                st.caption(meta)
            if audio_slot:
                with _bc:
                    if st.button("▶", key=f"hi_play_{_hi}", help=f"Play from {h['time_min']} min"):
                        _s = int(float(h["time_min"]) * 60)
                        st.session_state.audio_seek_time = _s
                        audio_slot.audio(audio_file_path, format=_audio_fmt,
                                         start_time=_s, autoplay=True)
    else:
        st.caption("None detected.")

with ki2:
    st.markdown("**Risk Signals**")
    if key_insights["risk_signals"]:
        for _ri, r in enumerate(key_insights["risk_signals"]):
            _raw = r.get("speaker", "")
            spk = _raw if _raw not in ("", "UNKNOWN") else _speaker_for_time(
                management_segments, r["time_min"], fallback_turns=av_turns, text=r["text"])
            st.error(r["text"])
            meta = " · ".join(x for x in [spk, f'{r["score"]:+.2f}', f'{r["time_min"]} min'] if x)
            _mc, _bc = st.columns([8, 1])
            with _mc:
                st.caption(meta)
            if audio_slot:
                with _bc:
                    if st.button("▶", key=f"rs_play_{_ri}", help=f"Play from {r['time_min']} min"):
                        _s = int(float(r["time_min"]) * 60)
                        st.session_state.audio_seek_time = _s
                        audio_slot.audio(audio_file_path, format=_audio_fmt,
                                         start_time=_s, autoplay=True)
    else:
        st.caption("None detected.")

st.divider()

# ================================================================
# SECTION 5: HISTORICAL TREND + DIVERGENCE
# ================================================================
hist_col, div_col2 = st.columns([1.6, 1])

with hist_col:
    st.subheader("Historical Sentiment Trend")
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
            line=dict(width=2), marker=dict(size=7),
        ))
        fig_t.add_trace(go.Scatter(
            x=qoq_df["label"], y=qoq_df["hedge_freq"],
            mode="lines+markers", name="Hedging /100w",
            line=dict(width=1.5, dash="dash"), marker=dict(size=5),
            yaxis="y2",
        ))
        fig_t.add_hline(y=0, line_width=1, line_dash="dot")
        fig_t.update_layout(
            height=230, margin=dict(l=0, r=40, t=10, b=0),
            yaxis=dict(title="Sentiment", range=[-1, 1]),
            yaxis2=dict(title="Hedging", overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(fig_t, use_container_width=True)
        hc1, hc2 = st.columns(2)
        hc1.metric("Sentiment vs prior Q", f'{current["sentiment"]:+.3f}', f'{d_sent:+.3f}')
        hc2.metric("Hedging vs prior Q", f'{current["hedge_freq"]:+.2f}', f'{d_hedge:+.2f}')
    else:
        st.info("Historical trend data could not be retrieved. Refresh and try running the analysis again.")

with div_col2:
    st.subheader("Tone-Text Divergence")
    if transcript_only or not audio_features.get("wav2vec2_raw_norm"):
        st.info("Audio required for tone-text divergence analysis.")
    else:
        section_div = multimodal_result.get("section_divergence", [])
        if section_div and is_full_call:
            sd_df = pd.DataFrame(section_div)
            fig_d = go.Figure()
            fig_d.add_trace(go.Bar(name="FinBERT", x=sd_df["section"], y=sd_df["sentiment"],
                                   marker_line_width=0))
            fig_d.add_trace(go.Bar(name="Wav2Vec2", x=sd_df["section"], y=sd_df["acoustic"],
                                   marker_line_width=0))
            fig_d.add_hline(y=0, line_width=1, line_dash="dot")
            fig_d.update_layout(barmode="group", height=230, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_d, use_container_width=True)
            st.caption("Gap = divergence. Acoustic below FinBERT = scripted positivity.")
        else:
            st.info("Section view requires full call audio (≥ 25 min).")

st.divider()

# ================================================================
# SECTION 6: SPEAKER ATTRIBUTION
# ================================================================
st.subheader("Speaker Attribution — Management MCI")

# Keep named management speakers; drop confirmed analysts/operators and
# unresolved pyannote labels (SPEAKER_XX / UNKNOWN).
spk_data = [
    s for s in insights["speaker_breakdown"]
    if s.get("speaker", "UNKNOWN") not in ("UNKNOWN", "")
    and not s.get("speaker", "").lower().startswith("speaker_")
    and is_management_speaker(s.get("speaker", "")) is not False
]

if not enriched_segments:
    st.info("Speaker attribution requires audio mode with Whisper transcription.")
elif not spk_data:
    st.info(
        "No named management speakers resolved. "
        "This occurs when pyannote diarization is unavailable or the Alpha Vantage "
        "transcript title regex did not match speaker introductions."
    )
else:
    sp_df = pd.DataFrame(spk_data)

    # Chart: MCI by speaker, split by section
    fig_sp = go.Figure()
    for sec in ["Prepared", "Q&A"]:
        sub = sp_df[sp_df["section"] == sec]
        if not sub.empty:
            fig_sp.add_trace(go.Bar(
                name=sec,
                x=sub["mci"],
                y=sub["speaker"],
                orientation="h",
                marker_line_width=0,
                text=[f"{v:.0f}" for v in sub["mci"]],
                textposition="outside",
            ))
    fig_sp.update_layout(
        barmode="group",
        height=max(180, len(sp_df) * 35),
        margin=dict(l=0, r=40, t=10, b=0),
        xaxis=dict(title="MCI [0–100]", range=[0, 110]),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_sp, use_container_width=True)
    st.caption("MCI = word count weighted FinBERT sentiment + audio features · Analysts and operators excluded")

st.divider()

# ================================================================
# SECTION 7: PEER COMPARISON
# ================================================================
peer_sector = insights.get("peer_sector", "Peer Group")
st.subheader(f"Peer Comparison — {peer_sector}")

peer_df = insights.get("peer_data", pd.DataFrame())
if not peer_df.empty and "is_selected" in peer_df.columns:
    display_df = peer_df[["rank", "ticker", "mci", "qa_stress", "signal", "is_selected"]].copy()
    display_df["ticker"] = display_df.apply(
        lambda r: f"● {r['ticker']}" if r["is_selected"] else r["ticker"], axis=1
    )
    display_df = display_df.rename(columns={
        "rank": "#", "ticker": "Ticker", "mci": "MCI",
        "qa_stress": "Q&A Stress", "signal": "Signal",
    }).drop(columns=["is_selected"])
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption(
        "MCI = FinBERT text sentiment [0–100] · "
        "Q&A Stress = sentiment decay in analyst Q&A · "
        "N/A = transcript not yet cached · ● = selected ticker"
    )
else:
    st.info("Peer data unavailable.")

st.divider()

# ================================================================
# SECTION 8: EVENT STUDY
# ================================================================
st.subheader("Event-Study Analysis — Earnings Sensitivity")
st.caption("CAPM event study: estimation window t−90 to t−20 · event window t−1 to t+3 · market proxy: SPY")

_es_col, _sec_col = st.columns([1.1, 1])

with _es_col:
    if "error" in es_result:
        st.warning(f"Event study unavailable: {es_result['error']}")
    else:
        _beta  = es_result["beta"]
        _car   = es_result["car"]
        _tstat = es_result["t_stat"]
        _rsq   = es_result["r_squared"]
        _n_est = es_result["n_est"]
        _edate = es_result["event_date"]
        _ar_df = es_result["ar_series"]
        _sig   = abs(_tstat) >= 1.96

        _ek1, _ek2, _ek3, _ek4 = st.columns([1, 1.2, 1, 0.9])
        _sens = "High" if _beta > 0.7 else "Medium" if _beta > 0.3 else "Low"
        _ek1.metric("Market Beta", f"{_beta:.2f}", help=f"General Market Sensitivity vs SPY · {_sens}")
        _ek2.metric("CAR [-1,+3]", f"{_car:+.2%}", help="Cumulative Abnormal Return")
        _ek3.metric("T-Stat", f"{_tstat:+.2f}",
                    help="Significant (95%)" if _sig else "Not significant")
        _ek4.metric("Model R²", f"{_rsq:.2f}",
                    help=f"{_n_est} est. days · t=0: {_edate}")

        fig_es = go.Figure()
        _bar_colours = ["green" if v >= 0 else "red" for v in _ar_df["ar"]]
        fig_es.add_trace(go.Bar(
            x=_ar_df["label"], y=_ar_df["ar"],
            name="Abnormal Return", marker_color=_bar_colours, marker_line_width=0,
        ))
        fig_es.add_trace(go.Scatter(
            x=_ar_df["label"], y=_ar_df["car_cumulative"],
            name="Cumulative AR", mode="lines+markers",
            line=dict(width=2), marker=dict(size=7),
        ))
        if "t=0" in list(_ar_df["label"].values):
            fig_es.add_vline(x=list(_ar_df["label"]).index("t=0"),
                             line_dash="dot", annotation_text="Event (t=0)")
        fig_es.update_layout(
            height=260, margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(tickformat=".1%", zeroline=True),
        )
        st.plotly_chart(fig_es, use_container_width=True)
        st.caption(
            f"Bars = daily AR (actual − CAPM expected). "
            f"Line = cumulative CAR. β = {_beta:.3f} from {_n_est}-day estimation window."
        )

with _sec_col:
    st.subheader("Earnings Price Sensitivity — Avg |CAR[−1,+3]|")
    _live_car          = es_result.get("car") if "error" not in es_result else None
    _peer_sector_label = insights.get("peer_sector")
    _eps_data          = compute_sector_earnings_sensitivity()
    _sec_df            = get_sector_sensitivity_df(_eps_data, _peer_sector_label, _live_car)

    fig_sec = go.Figure()
    fig_sec.add_trace(go.Bar(
        x=_sec_df["eps"], y=_sec_df["sector"], orientation="h",
        marker_color=_sec_df["colour"].tolist(), marker_line_width=0,
        text=[f"{v*100:.1f}%" for v in _sec_df["eps"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Avg |CAR| = %{x:.1%}<extra></extra>",
    ))
    for _thresh, _lbl in [(0.03, "Low | Med"), (0.06, "Med | High")]:
        fig_sec.add_vline(x=_thresh, line_dash="dot",
                          annotation_text=_lbl, annotation_font_size=9)
    fig_sec.update_layout(
        height=520, margin=dict(l=0, r=40, t=10, b=0), showlegend=False,
        xaxis=dict(title="Avg |CAR| — event window [−1, +3]",
                   range=[0, max(_sec_df["eps"]) * 1.25], tickformat=".0%"),
    )
    st.plotly_chart(fig_sec, use_container_width=True)
    _eps_source = "Live computed" if _eps_data else "Reference fallback"
    st.caption(
        f"🟢 Low (<3%) · 🟡 Medium (3–6%) · "
        f"🔴 High (>6%) · 🔵 Selected sector (live |CAR|) · Source: {_eps_source}"
    )

st.divider()

# ================================================================
# SECTION 9: METHODOLOGY
# ================================================================
st.subheader("How EarningsIQ Analyses Calls")

with st.expander("Data sources & ingestion", expanded=False):
    st.markdown("""
**Audio** files are sourced from [EarningsCallBiz](https://www.earningscall.biz) and
placed in the `cache/` directory. The pipeline reads them directly from there 
no re-download occurs on repeat runs. If no cached file is found for a ticker/quarter,
yt-dlp attempts a YouTube fallback; if that also fails (common on cloud hosts),
the pipeline degrades gracefully to transcript-only mode.

**Transcripts** come from Alpha Vantage's earnings call API. They are used for:
- Speaker name resolution (mapping pyannote's SPEAKER_00/01 labels to real names and titles)
- Peer comparison (computing MCI for competitor firms without processing audio)
- Historical QoQ sentiment (fetching the previous 6 quarters to compute the Narrative Shift Index)

If no transcript is available, speaker names fall back to pyannote's generic SPEAKER_XX labels.
Charts requiring audio (trajectory, divergence, speaker attribution) show an info message
when operating in transcript-only mode.
""")

with st.expander("Transcript analysis — how sentiment is measured", expanded=False):
    st.markdown("""
**Model**: [FinBERT](https://huggingface.co/ProsusAI/finbert) — a BERT model fine-tuned on
financial text (earnings call transcripts, SEC filings, financial news). It outperforms
general purpose sentiment models on earnings call language because it understands
domain specific phrases like "headwinds", "beat consensus", and "margin expansion".

**Segment level scoring**: Whisper splits the audio into natural speech segments
(typically 5–20 seconds each). FinBERT scores each segment independently, producing
a timestamped sentiment trajectory rather than a single bulk score. This powers the
intra-call timeline chart.

**Word count weighting**: When aggregating across segments, each segment is weighted
by its word count. Filler phrases ("Thank you", "That's a great question") carry
proportionally less weight than 100-word explanations of operating leverage.
Segments under 8 words are excluded from aggregates entirely.

**Multi sample for history**: When analysing previous quarters for QoQ comparison,
5 evenly spaced 1,500 character windows are sampled across the transcript, skipping
the first 10% (safe harbour boilerplate). Their scores are averaged for a fair
cross quarter comparison.

**Sentiment score**: `positive_probability − negative_probability`, range [−1, +1].
Zero = neutral; +1 = maximally positive; −1 = maximally negative.
Typical earnings calls land between +0.1 and +0.5.
""")

with st.expander("Audio analysis — Wav2Vec2 and acoustic features", expanded=False):
    st.markdown("""
**Transcription**: [OpenAI Whisper](https://github.com/openai/whisper) (base model)
converts the audio to text with timestamps. It runs locally on CPU (or Apple MPS).
Results are cached after the first run.

**Speaker diarization**: [pyannote](https://github.com/pyannote/pyannote-audio) 3.1
assigns a speaker label (SPEAKER_00, SPEAKER_01, …) to each audio segment. Labels are
then resolved to real names using word-overlap matching against the Alpha Vantage
transcript. Analysts and IR representatives are filtered out; only named management
speakers (CEO, CFO, COO, etc.) contribute to MCI and sentiment scores.

**Acoustic features** (librosa):
- *Pitch mean & std* — sampled across start, middle, and end of the call. High pitch
  coefficient of variation indicates vocal instability.
- *Pause ratio* — frames where RMS energy drops below 20% of the call mean.
  Compressed audio from YouTube raises the noise floor, so 20% is used rather than 10%.
- *Tempo* — speaking rate in BPM.

**Wav2Vec2** ([facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)):
A transformer trained on 960 hours of LibriSpeech speech. Three windows are sampled
(start, middle, end) to cover the full call without exceeding memory limits.
The confidence proxy is derived from the **temporal coefficient of variation** of
per timestep embedding norms. More dynamic, engaged speech produces higher variation
than monotone scripted delivery. This is codec independent (unlike the raw norm).
""")

with st.expander("How MCI is calculated", expanded=False):
    st.markdown("""
**Management Confidence Index (MCI)** is a composite score from 0 to 100.

| Component | Weight | Source | Interpretation |
|-----------|--------|---------|----------------|
| FinBERT net sentiment | 40% | Text | What management *says* |
| Wav2Vec2 confidence proxy | 35% | Audio | How management *sounds* |
| Inverse pause ratio | 15% | Audio | Hesitation / fluency |
| Inverse pitch CV | 10% | Audio | Vocal steadiness |

Weights follow Chen et al. (2023): audio adds incremental value beyond text
but text remains the primary signal for earnings calls.

**Formula**: `MCI = (sentiment_norm × 0.40 + wav2vec2 × 0.35 + pause_inv × 0.15 + pitch_inv × 0.10) × 100`

where `sentiment_norm = (FinBERT_score + 1) / 2` maps [−1, +1] → [0, 1].

**Tone-text divergence** = `(Wav2Vec2 − FinBERT_norm) × 0.5`

Negative divergence means acoustic confidence is below textual sentiment, a pattern
associated with scripted positivity masking genuine hesitation (Hajek & Munk, 2023).
Flag threshold: −0.12.
""")

with st.expander("Narrative Shift Index (NSI) and Q&A Stress", expanded=False):
    st.markdown("""
**NSI** measures how far the current call's language has shifted from the firm's
own historical baseline across the prior 6 quarters.

`NSI σ = (current_sentiment − historical_mean) / historical_std`

A floor of 0.05 is applied to `historical_std` to prevent absurd sigma values when
prior quarters had very similar transcripts (near-zero variance). Typical range: ±3σ.
Interpretation: +2σ = unusually bullish vs history; −2σ = unusually cautious.

**Q&A Stress** = prepared remarks sentiment − Q&A sentiment (word-count weighted).

The Q&A section start is detected by scanning Whisper segments for operator
transition phrases ("your first question", "we'll now open the floor", etc.).
Falls back to the 65% duration heuristic only when no phrase is found.
Price et al. (2012) show the Q&A section has the highest alpha among call sections
because it is unscripted and analysts ask pointed questions.

Threshold: >0.20 = high stress; >0.10 = moderate.
""")

with st.expander("Event study — Cumulative Abnormal Return (CAR)", expanded=False):
    st.markdown("""
Methodology: CAPM event study following Brown & Warner (1985).

**Estimation window**: t = −120 to t = −20 trading days before the earnings date.
OLS regression of stock returns on SPY returns estimates the stock's normal return
behaviour: `R_stock = α + β × R_SPY`.

**Event window**: t = −1 to t = +3 trading days.
Abnormal return on each day: `AR_t = R_actual_t − (α + β × R_SPY_t)`.
Cumulative abnormal return: `CAR = Σ AR_t`.

**T-statistic**: `CAR / (σ_est × √n_event)` where σ_est is the residual standard
deviation from the estimation window. Significant at 95% if |t| ≥ 1.96.

**Market Beta (β)** shown in the metrics is the general CAPM beta from the estimation
window — it measures normal day-to-day market sensitivity, not earnings-specific
sensitivity. The earnings-specific measure is **CAR [−1,+3]**.

**Earnings Price Sensitivity** chart shows the average |CAR[−1,+3]| across the last
3 earnings events for 3 representative tickers per sector, computed live from
yfinance price data using calendar-approximate earnings dates.
""")
