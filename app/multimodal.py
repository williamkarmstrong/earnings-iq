"""
Multimodal analysis module.
Fuses text sentiment (FinBERT) with audio features (librosa + Wav2Vec2)
to produce the divergence scores shown in the dashboard.

The core insight: when acoustic confidence diverges from textual sentiment,
it signals that management delivery does not match their words —
a pattern Hajek & Munk (2023) show has predictive value for financial outcomes.
"""

import numpy as np
import pandas as pd


def compute_text_mci(sentiment_score):
    """
    Text-only MCI — FinBERT sentiment mapped to [0, 100].
    Used for peer comparison so all values are on the same basis
    (illustrative peer MCIs are text-derived; mixing in audio for the
    selected ticker creates an apples-to-oranges comparison).
    """
    return round(float(((sentiment_score + 1) / 2) * 100), 1)


def compute_mci(sentiment_score, audio_features):
    """
    Management Confidence Index — composite 0-100 score.
    Weighted combination of:
      - FinBERT net sentiment (40%) — what is said
      - Wav2Vec2 confidence proxy (35%) — how it is delivered
      - Pause ratio inverted (15%) — hesitation penalty
      - Pitch stability inverted (10%) — vocal steadiness
    Weights reflect Chen et al. (2023): audio adds incremental value
    beyond text but text remains the primary signal.
    """
    text_component  = (sentiment_score + 1) / 2
    deep_component  = audio_features.get("confidence_proxy", 0.5)
    pause_component = 1.0 - min(audio_features.get("pause_ratio", 0.3), 1.0)

    pitch_mean = audio_features.get("pitch_mean", 150)
    pitch_std  = audio_features.get("pitch_std", 30)
    if pitch_mean > 0:
        cv = pitch_std / pitch_mean
        pitch_component = 1.0 - min(cv, 1.0)
    else:
        pitch_component = 0.5

    mci = (
        text_component  * 0.40 +
        deep_component  * 0.35 +
        pause_component * 0.15 +
        pitch_component * 0.10
    ) * 100

    return round(float(mci), 1)


def compute_tone_text_divergence(sentiment_score, audio_features):
    """
    Tone-text divergence — the key proprietary signal.
    Measures the gap between what is said (FinBERT) and how it is delivered
    (Wav2Vec2). Negative values mean acoustic confidence is below textual
    sentiment — indicating scripted positivity masking genuine hesitation.
    Score range approximately [-1, +1].
    """
    text_normalised     = (sentiment_score + 1) / 2
    acoustic_normalised = audio_features.get("confidence_proxy", 0.5)
    # Scale by 0.5: earnings call language is structurally positive (text_normalised
    # clusters at 0.65-0.85) while audio proxy sits near 0.5, creating a raw gap of
    # -0.2 to -0.4 on every call. Halving compresses to a range where -0.12 genuinely
    # flags divergence rather than routine language-delivery skew.
    divergence          = (acoustic_normalised - text_normalised) * 0.5
    return round(float(divergence), 3)


def compute_segment_timeline(enriched_segments, audio_features):
    """
    Build the intra-call timeline from per-segment FinBERT scores.
    The acoustic_confidence column now holds the global Wav2Vec2 proxy
    as a flat reference line -- useful context without misleading per-segment noise.
    Returns a DataFrame with columns: time_min, sentiment_score,
    acoustic_confidence, speaker, section.
    """
    if not enriched_segments:
        return pd.DataFrame()

    global_proxy = audio_features.get("confidence_proxy", 0.5)
    # Normalise proxy to [-1,+1] to align with FinBERT axis
    proxy_normalised = round(global_proxy * 2 - 1, 3)
    max_time = max(s.get("end", 0) for s in enriched_segments)
    qa_start = max_time * 0.65

    rows = []
    for seg in enriched_segments:
        section = "Q&A" if seg.get("start", 0) >= qa_start else "Prepared"
        rows.append({
            "time_min":            round(seg.get("start", 0) / 60, 2),
            "sentiment_score":     round(seg.get("sentiment_score", 0.0), 3),
            "acoustic_confidence": proxy_normalised,  # flat reference line
            "speaker":             seg.get("speaker", "UNKNOWN"),
            "section":             section,
        })

    return pd.DataFrame(rows)


def compute_section_divergence(enriched_segments, audio_features):
    """
    Section-level tone-text divergence: compare avg FinBERT sentiment
    in Prepared vs Q&A sections against the global Wav2Vec2 proxy.
    Returns a list of dicts suitable for a bar chart.
    """
    if not enriched_segments:
        return []

    global_proxy = audio_features.get("confidence_proxy", 0.5)
    proxy_norm   = global_proxy * 2 - 1  # to [-1,+1] scale

    max_time = max(s.get("end", 0) for s in enriched_segments)
    qa_start = max_time * 0.65

    prep_scores = [s.get("sentiment_score", 0.0) for s in enriched_segments if s.get("start", 0) < qa_start]
    qa_scores   = [s.get("sentiment_score", 0.0) for s in enriched_segments if s.get("start", 0) >= qa_start]

    rows = []
    for label, scores in [("Prepared", prep_scores), ("Q&A", qa_scores)]:
        if scores:
            avg_sent = float(np.mean(scores))
            rows.append({
                "section":    label,
                "sentiment":  round(avg_sent, 3),
                "acoustic":   round(proxy_norm, 3),
                "divergence": round(proxy_norm - avg_sent, 3),
            })
    return rows


def compute_speaker_mci(enriched_segments, audio_features):
    """
    Break MCI down by speaker and section (prepared vs. Q&A).
    Uses the same weighting as compute_mci applied to per-speaker
    average sentiment, with global audio features shared.
    Returns a list of dicts for the speaker attribution panel.
    """
    from collections import defaultdict

    max_time = max((s.get("end", 0) for s in enriched_segments), default=0)
    qa_start = max_time * 0.65

    groups = defaultdict(list)
    for seg in enriched_segments:
        section = "Q&A" if seg.get("start", 0) >= qa_start else "Prepared"
        key = (seg.get("speaker", "UNKNOWN"), section)
        groups[key].append(seg.get("sentiment_score", 0.0))

    results = []
    for (speaker, section), scores in groups.items():
        avg_sentiment = float(np.mean(scores))
        mci = compute_mci(avg_sentiment, audio_features)
        results.append({
            "speaker":       speaker,
            "section":       section,
            "mci":           mci,
            "avg_sentiment": round(avg_sentiment, 3),
        })

    results.sort(key=lambda x: (x["section"], x["speaker"]))
    return results


def analyse_multimodal(text_sentiment, audio_features, enriched_segments=None):
    """
    Main entry point — runs all multimodal computations and returns
    a single result dict that insights.py and app.py consume.
    """
    mci        = compute_mci(text_sentiment, audio_features)
    text_mci   = compute_text_mci(text_sentiment)
    divergence = compute_tone_text_divergence(text_sentiment, audio_features)

    result = {
        "mci":                mci,
        "text_mci":           text_mci,
        "tone_text_divergence": divergence,
        "timeline":           pd.DataFrame(),
        "speaker_breakdown":  [],
    }

    if enriched_segments:
        result["timeline"]           = compute_segment_timeline(enriched_segments, audio_features)
        result["speaker_breakdown"]  = compute_speaker_mci(enriched_segments, audio_features)
        result["section_divergence"] = compute_section_divergence(enriched_segments, audio_features)

    return result
