"""
Insight generation module.
Turns raw multimodal scores into dashboard-ready signals and flags.
Peer benchmarking uses mock data for the demo — will be replaced
with cached real call data in the next iteration.
"""

import pandas as pd
import streamlit as st

# Signal thresholds: Low ≤52 | Medium 52–56 | High ≥56
SIGNAL_MCI_POSITIVE = 56   # text_mci >= this → "Positive"
SIGNAL_MCI_WATCH    = 52   # text_mci <= this → "Watch", else "Neutral"

MCI_THRESHOLDS = {
    "High":   (56, 100),  # checked first — 56 resolves to High, not Medium
    "Medium": (52,  56),  # 52 resolves to Medium, not Low
    "Low":    (0,   52),
}

DIVERGENCE_THRESHOLDS = {
    "Positive":  ( 0.05,  1.0),
    "Neutral":   (-0.05,  0.05),
    "Elevated":  (-1.0,  -0.05),
}


def _label_mci(mci):
    """Map MCI score to a text label."""
    for label, (lo, hi) in MCI_THRESHOLDS.items():
        if lo <= mci <= hi:
            return label
    return "Unknown"


def _label_divergence(div):
    """Map divergence score to a text label."""
    for label, (lo, hi) in DIVERGENCE_THRESHOLDS.items():
        if lo <= div <= hi:
            return label
    return "Unknown"


def generate_signal_flags(multimodal_result, hedge_data, prepared_sentiment, qa_sentiment):
    """
    Generate the prioritised analyst flag list for the signal panel.
    Each flag has severity (high/medium/low), message, and attribution.
    Ordered high to low so the most actionable items appear first.
    """
    flags = []
    mci        = multimodal_result.get("mci", 50)
    divergence = multimodal_result.get("tone_text_divergence", 0)

    # MCI flags
    if mci < 52:
        flags.append({"severity": "high",
                      "message": f"Management confidence low at {mci}/100",
                      "attribution": "Composite score - full call"})
    elif mci < 56:
        flags.append({"severity": "medium",
                      "message": f"Management confidence moderate at {mci}/100",
                      "attribution": "Composite score - full call"})
    else:
        flags.append({"severity": "low",
                      "message": f"Management confidence positive at {mci}/100",
                      "attribution": "Composite score - full call"})

    # Tone-text divergence flags (scale: ±0.5 after halving raw gap)
    if divergence < -0.12:
        flags.append({"severity": "high",
                      "message": f"Tone-text divergence elevated at {divergence:+.2f} - acoustic confidence below textual sentiment",
                      "attribution": "Multimodal - full call"})
    elif divergence < -0.05:
        flags.append({"severity": "medium",
                      "message": f"Moderate tone-text divergence ({divergence:+.2f})",
                      "attribution": "Multimodal - full call"})

    # Q&A sentiment decay — Price et al. (2012) identify this as highest-alpha split
    if prepared_sentiment is not None and qa_sentiment is not None:
        decay = prepared_sentiment - qa_sentiment
        if decay > 0.20:
            flags.append({"severity": "high",
                          "message": f"Q&A sentiment decay of {decay:.2f} - significant drop from prepared remarks",
                          "attribution": "Q&A section - analyst questioning phase"})
        elif decay > 0.10:
            flags.append({"severity": "medium",
                          "message": f"Moderate Q&A sentiment decay ({decay:.2f})",
                          "attribution": "Q&A section"})
        else:
            flags.append({"severity": "low",
                          "message": "Sentiment stable across prepared remarks and Q&A",
                          "attribution": "Section comparison"})

    # Hedging language flag
    hedge_freq = hedge_data.get("frequency_per_100", 0)
    if hedge_freq > 8:
        flags.append({"severity": "medium",
                      "message": f"High hedging language frequency: {hedge_freq:.1f} instances per 100 words",
                      "attribution": "NLP - full transcript"})

    # Positive signal if nothing high severity
    if not any(f["severity"] == "high" for f in flags):
        flags.append({"severity": "low",
                      "message": "No high-priority signals detected - tone broadly consistent with text",
                      "attribution": "Composite assessment"})

    return flags


# Sector peer groups -- tickers only; metrics computed live from real transcripts.
# Tickers limited to names with Alpha Vantage earnings transcript coverage.
_PEER_UNIVERSE = {
    "mega_tech": {
        "label": "Mega-cap Tech",
        "data": [
            {"ticker": "AAPL"},
            {"ticker": "MSFT"},
            {"ticker": "GOOGL"},
            {"ticker": "GOOG"},
            {"ticker": "META"},
            {"ticker": "AMZN"},
            {"ticker": "ORCL"},
        ],
    },
    "semiconductors": {
        "label": "Semiconductors",
        "data": [
            {"ticker": "NVDA"},
            {"ticker": "AMD"},
            {"ticker": "INTC"},
            {"ticker": "QCOM"},
            {"ticker": "AVGO"},
            {"ticker": "MU"},
            {"ticker": "TXN"},
            {"ticker": "AMAT"},
        ],
    },
    "ev_auto": {
        "label": "EV & Auto",
        "data": [
            {"ticker": "TSLA"},
            {"ticker": "F"},
            {"ticker": "GM"},
            {"ticker": "RIVN"},
            {"ticker": "NIO"},
            {"ticker": "STLA"},
        ],
    },
    "cloud_saas": {
        "label": "Cloud & SaaS",
        "data": [
            {"ticker": "CRM"},
            {"ticker": "NOW"},
            {"ticker": "SNOW"},
            {"ticker": "DDOG"},
            {"ticker": "ZM"},
            {"ticker": "OKTA"},
            {"ticker": "MDB"},
            {"ticker": "TEAM"},
        ],
    },
    "cybersecurity": {
        "label": "Cybersecurity",
        "data": [
            {"ticker": "CRWD"},
            {"ticker": "PANW"},
            {"ticker": "FTNT"},
            {"ticker": "ZS"},
            {"ticker": "S"},
            {"ticker": "CYBR"},
        ],
    },
    "streaming_media": {
        "label": "Streaming & Media",
        "data": [
            {"ticker": "NFLX"},
            {"ticker": "DIS"},
            {"ticker": "WBD"},
            {"ticker": "PARA"},
            {"ticker": "SPOT"},
            {"ticker": "ROKU"},
        ],
    },
    "social_media": {
        "label": "Social Media",
        "data": [
            {"ticker": "SNAP"},
            {"ticker": "PINS"},
            {"ticker": "RDDT"},
            {"ticker": "MTCH"},
            {"ticker": "TTD"},
        ],
    },
    "ecommerce": {
        "label": "E-commerce",
        "data": [
            {"ticker": "SHOP"},
            {"ticker": "EBAY"},
            {"ticker": "ETSY"},
            {"ticker": "W"},
            {"ticker": "BABA"},
            {"ticker": "JD"},
        ],
    },
    "banks": {
        "label": "Major Banks",
        "data": [
            {"ticker": "JPM"},
            {"ticker": "BAC"},
            {"ticker": "WFC"},
            {"ticker": "C"},
            {"ticker": "GS"},
            {"ticker": "MS"},
            {"ticker": "USB"},
        ],
    },
    "fintech_payments": {
        "label": "Fintech & Payments",
        "data": [
            {"ticker": "V"},
            {"ticker": "MA"},
            {"ticker": "PYPL"},
            {"ticker": "SQ"},
            {"ticker": "AFRM"},
            {"ticker": "SOFI"},
            {"ticker": "NU"},
        ],
    },
    "pharma": {
        "label": "Pharmaceuticals",
        "data": [
            {"ticker": "JNJ"},
            {"ticker": "PFE"},
            {"ticker": "ABBV"},
            {"ticker": "MRK"},
            {"ticker": "LLY"},
            {"ticker": "BMY"},
            {"ticker": "AZN"},
        ],
    },
    "biotech": {
        "label": "Biotech",
        "data": [
            {"ticker": "MRNA"},
            {"ticker": "REGN"},
            {"ticker": "BIIB"},
            {"ticker": "GILD"},
            {"ticker": "AMGN"},
            {"ticker": "VRTX"},
        ],
    },
    "healthtech": {
        "label": "Healthcare Tech",
        "data": [
            {"ticker": "ISRG"},
            {"ticker": "DXCM"},
            {"ticker": "TDOC"},
            {"ticker": "VEEV"},
            {"ticker": "IDXX"},
            {"ticker": "ILMN"},
        ],
    },
    "energy": {
        "label": "Energy",
        "data": [
            {"ticker": "XOM"},
            {"ticker": "CVX"},
            {"ticker": "COP"},
            {"ticker": "SLB"},
            {"ticker": "EOG"},
            {"ticker": "PSX"},
            {"ticker": "VLO"},
        ],
    },
    "consumer_staples": {
        "label": "Consumer Staples",
        "data": [
            {"ticker": "PG"},
            {"ticker": "KO"},
            {"ticker": "PEP"},
            {"ticker": "WMT"},
            {"ticker": "COST"},
            {"ticker": "CL"},
            {"ticker": "MO"},
        ],
    },
    "consumer_discretionary": {
        "label": "Consumer Discretionary",
        "data": [
            {"ticker": "NKE"},
            {"ticker": "MCD"},
            {"ticker": "SBUX"},
            {"ticker": "HD"},
            {"ticker": "TGT"},
            {"ticker": "LOW"},
            {"ticker": "BKNG"},
        ],
    },
    "industrial": {
        "label": "Industrial",
        "data": [
            {"ticker": "CAT"},
            {"ticker": "HON"},
            {"ticker": "GE"},
            {"ticker": "MMM"},
            {"ticker": "EMR"},
            {"ticker": "ETN"},
            {"ticker": "PH"},
        ],
    },
    "aerospace_defense": {
        "label": "Aerospace & Defense",
        "data": [
            {"ticker": "BA"},
            {"ticker": "LMT"},
            {"ticker": "RTX"},
            {"ticker": "NOC"},
            {"ticker": "GD"},
            {"ticker": "HII"},
        ],
    },
    "telecom": {
        "label": "Telecom",
        "data": [
            {"ticker": "T"},
            {"ticker": "VZ"},
            {"ticker": "TMUS"},
            {"ticker": "CHTR"},
            {"ticker": "LUMN"},
            {"ticker": "CMCSA"},
        ],
    },
    "ai_data": {
        "label": "AI & Data Analytics",
        "data": [
            {"ticker": "PLTR"},
            {"ticker": "IBM"},
            {"ticker": "AI"},
            {"ticker": "PATH"},
            {"ticker": "SMAR"},
            {"ticker": "BBAI"},
        ],
    },
}

# Flat lookup: ticker -> sector key
_TICKER_SECTOR = {
    item["ticker"]: key
    for key, group in _PEER_UNIVERSE.items()
    for item in group["data"]
}


def _get_sector(ticker):
    """Return (sector_key, sector_label) for a ticker."""
    key = _TICKER_SECTOR.get(ticker.upper(), "mega_tech")
    return key, _PEER_UNIVERSE[key]["label"]


def get_peer_tickers(ticker):
    """
    Return (peer_ticker_list, sector_label) for the ticker's sector.
    The returned list includes the selected ticker itself.
    """
    ticker_upper = ticker.upper()
    sector_key, sector_label = _get_sector(ticker_upper)
    tickers = [item["ticker"] for item in _PEER_UNIVERSE[sector_key]["data"]]
    return tickers, sector_label

@st.cache_data(show_spinner=False)
def generate_insights(multimodal_result, hedge_data, prepared_sentiment, qa_sentiment, ticker):
    """
    Main entry point — assembles the full insights package that app.py consumes.
    Returns a single dict with everything the dashboard needs.
    """
    mci        = multimodal_result.get("mci", 50)
    divergence = multimodal_result.get("tone_text_divergence", 0)

    return {
        "mci":                mci,
        "mci_label":          _label_mci(mci),
        "tone_text_divergence": divergence,
        "divergence_label":   _label_divergence(divergence),
        "qa_decay":           round((prepared_sentiment or 0) - (qa_sentiment or 0), 3),
        "qa_stress":          "High"     if (prepared_sentiment or 0) - (qa_sentiment or 0) > 0.20 else
                              "Moderate" if (prepared_sentiment or 0) - (qa_sentiment or 0) > 0.10 else "Low",
        "hedge_frequency":    hedge_data.get("frequency_per_100", 0),
        "flags":              generate_signal_flags(multimodal_result, hedge_data, prepared_sentiment, qa_sentiment),
        "timeline":           multimodal_result.get("timeline", pd.DataFrame()),
        "speaker_breakdown":  multimodal_result.get("speaker_breakdown", []),
        **dict(zip(("peer_tickers","peer_sector"), get_peer_tickers(ticker))),
    }
