"""
Insight generation module.
Turns raw multimodal scores into dashboard-ready signals and flags.
Peer benchmarking uses mock data for the demo — will be replaced
with cached real call data in the next iteration.
"""

import pandas as pd


# Signal thresholds — calibrated to the concept paper's NVDA case example
# (MCI=61, divergence=-0.38) as a reference point.
MCI_THRESHOLDS = {
    "High":     (75, 100),
    "Moderate": (55, 75),
    "Low":      (0,  55),
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
    if mci < 55:
        flags.append({"severity": "high",
                      "message": f"Management confidence index low at {mci}/100",
                      "attribution": "Composite score - full call"})
    elif mci < 70:
        flags.append({"severity": "medium",
                      "message": f"Management confidence moderate at {mci}/100",
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


# Sector peer groups -- illustrative data, replace with cached real-call analysis.
# Tickers limited to names with Alpha Vantage earnings transcript coverage.
_PEER_UNIVERSE = {
    "mega_tech": {
        "label": "Mega-cap Tech",
        "data": [
            {"ticker": "AAPL",  "mci": 74, "divergence": -0.12, "signal": "Neutral",  "nsi_sigma":  0.4, "qa_stress": 0.08},
            {"ticker": "MSFT",  "mci": 82, "divergence":  0.15, "signal": "Positive", "nsi_sigma": -0.3, "qa_stress": 0.05},
            {"ticker": "GOOGL", "mci": 71, "divergence": -0.09, "signal": "Neutral",  "nsi_sigma":  0.2, "qa_stress": 0.10},
            {"ticker": "GOOG",  "mci": 71, "divergence": -0.09, "signal": "Neutral",  "nsi_sigma":  0.2, "qa_stress": 0.10},
            {"ticker": "META",  "mci": 78, "divergence":  0.09, "signal": "Positive", "nsi_sigma": -0.1, "qa_stress": 0.06},
            {"ticker": "AMZN",  "mci": 76, "divergence":  0.06, "signal": "Positive", "nsi_sigma": -0.2, "qa_stress": 0.07},
            {"ticker": "ORCL",  "mci": 70, "divergence": -0.07, "signal": "Neutral",  "nsi_sigma":  0.3, "qa_stress": 0.11},
        ],
    },
    "semiconductors": {
        "label": "Semiconductors",
        "data": [
            {"ticker": "NVDA",  "mci": 61, "divergence": -0.38, "signal": "Watch",    "nsi_sigma":  2.1, "qa_stress": 0.22},
            {"ticker": "AMD",   "mci": 69, "divergence": -0.11, "signal": "Neutral",  "nsi_sigma":  0.5, "qa_stress": 0.12},
            {"ticker": "INTC",  "mci": 52, "divergence": -0.18, "signal": "Watch",    "nsi_sigma": -0.8, "qa_stress": 0.18},
            {"ticker": "QCOM",  "mci": 72, "divergence": -0.05, "signal": "Neutral",  "nsi_sigma":  0.3, "qa_stress": 0.09},
            {"ticker": "AVGO",  "mci": 80, "divergence":  0.11, "signal": "Positive", "nsi_sigma": -0.4, "qa_stress": 0.06},
            {"ticker": "MU",    "mci": 64, "divergence": -0.14, "signal": "Neutral",  "nsi_sigma":  0.9, "qa_stress": 0.14},
            {"ticker": "TXN",   "mci": 77, "divergence":  0.08, "signal": "Positive", "nsi_sigma": -0.2, "qa_stress": 0.07},
            {"ticker": "AMAT",  "mci": 73, "divergence": -0.04, "signal": "Neutral",  "nsi_sigma":  0.2, "qa_stress": 0.10},
        ],
    },
    "ev_auto": {
        "label": "EV & Auto",
        "data": [
            {"ticker": "TSLA",  "mci": 58, "divergence": -0.21, "signal": "Watch",    "nsi_sigma":  1.2, "qa_stress": 0.21},
            {"ticker": "F",     "mci": 63, "divergence": -0.08, "signal": "Neutral",  "nsi_sigma":  0.3, "qa_stress": 0.13},
            {"ticker": "GM",    "mci": 67, "divergence": -0.06, "signal": "Neutral",  "nsi_sigma": -0.1, "qa_stress": 0.11},
            {"ticker": "RIVN",  "mci": 48, "divergence": -0.31, "signal": "Watch",    "nsi_sigma":  1.8, "qa_stress": 0.26},
            {"ticker": "NIO",   "mci": 44, "divergence": -0.42, "signal": "Watch",    "nsi_sigma":  2.3, "qa_stress": 0.30},
            {"ticker": "STLA",  "mci": 65, "divergence": -0.05, "signal": "Neutral",  "nsi_sigma": -0.1, "qa_stress": 0.10},
        ],
    },
    "cloud_saas": {
        "label": "Cloud & SaaS",
        "data": [
            {"ticker": "CRM",   "mci": 75, "divergence":  0.05, "signal": "Positive", "nsi_sigma": -0.2, "qa_stress": 0.07},
            {"ticker": "NOW",   "mci": 79, "divergence":  0.12, "signal": "Positive", "nsi_sigma": -0.5, "qa_stress": 0.05},
            {"ticker": "SNOW",  "mci": 66, "divergence": -0.09, "signal": "Neutral",  "nsi_sigma":  0.7, "qa_stress": 0.13},
            {"ticker": "DDOG",  "mci": 73, "divergence": -0.04, "signal": "Neutral",  "nsi_sigma":  0.2, "qa_stress": 0.09},
            {"ticker": "ZM",    "mci": 55, "divergence": -0.19, "signal": "Watch",    "nsi_sigma":  0.4, "qa_stress": 0.17},
            {"ticker": "OKTA",  "mci": 61, "divergence": -0.13, "signal": "Watch",    "nsi_sigma":  0.6, "qa_stress": 0.16},
            {"ticker": "MDB",   "mci": 70, "divergence": -0.06, "signal": "Neutral",  "nsi_sigma":  0.1, "qa_stress": 0.10},
            {"ticker": "TEAM",  "mci": 72, "divergence": -0.03, "signal": "Neutral",  "nsi_sigma": -0.3, "qa_stress": 0.08},
        ],
    },
    "cybersecurity": {
        "label": "Cybersecurity",
        "data": [
            {"ticker": "CRWD",  "mci": 81, "divergence":  0.14, "signal": "Positive", "nsi_sigma": -0.4, "qa_stress": 0.04},
            {"ticker": "PANW",  "mci": 77, "divergence":  0.08, "signal": "Positive", "nsi_sigma": -0.3, "qa_stress": 0.06},
            {"ticker": "FTNT",  "mci": 74, "divergence": -0.03, "signal": "Neutral",  "nsi_sigma":  0.1, "qa_stress": 0.09},
            {"ticker": "ZS",    "mci": 72, "divergence": -0.05, "signal": "Neutral",  "nsi_sigma":  0.2, "qa_stress": 0.10},
            {"ticker": "S",     "mci": 63, "divergence": -0.11, "signal": "Neutral",  "nsi_sigma":  0.5, "qa_stress": 0.14},
            {"ticker": "CYBR",  "mci": 68, "divergence": -0.08, "signal": "Neutral",  "nsi_sigma":  0.3, "qa_stress": 0.12},
        ],
    },
    "streaming_media": {
        "label": "Streaming & Media",
        "data": [
            {"ticker": "NFLX",  "mci": 76, "divergence":  0.07, "signal": "Positive", "nsi_sigma": -0.3, "qa_stress": 0.07},
            {"ticker": "DIS",   "mci": 62, "divergence": -0.16, "signal": "Neutral",  "nsi_sigma":  0.4, "qa_stress": 0.15},
            {"ticker": "WBD",   "mci": 54, "divergence": -0.22, "signal": "Watch",    "nsi_sigma":  0.9, "qa_stress": 0.20},
            {"ticker": "PARA",  "mci": 49, "divergence": -0.28, "signal": "Watch",    "nsi_sigma":  1.1, "qa_stress": 0.24},
            {"ticker": "SPOT",  "mci": 68, "divergence": -0.07, "signal": "Neutral",  "nsi_sigma":  0.2, "qa_stress": 0.11},
            {"ticker": "ROKU",  "mci": 59, "divergence": -0.17, "signal": "Watch",    "nsi_sigma":  0.7, "qa_stress": 0.16},
        ],
    },
    "social_media": {
        "label": "Social Media",
        "data": [
            {"ticker": "SNAP",  "mci": 57, "divergence": -0.20, "signal": "Watch",    "nsi_sigma":  0.8, "qa_stress": 0.19},
            {"ticker": "PINS",  "mci": 65, "divergence": -0.10, "signal": "Neutral",  "nsi_sigma":  0.3, "qa_stress": 0.12},
            {"ticker": "RDDT",  "mci": 63, "divergence": -0.09, "signal": "Neutral",  "nsi_sigma":  0.2, "qa_stress": 0.11},
            {"ticker": "MTCH",  "mci": 60, "divergence": -0.14, "signal": "Watch",    "nsi_sigma":  0.5, "qa_stress": 0.15},
            {"ticker": "TTD",   "mci": 74, "divergence":  0.03, "signal": "Neutral",  "nsi_sigma":  0.1, "qa_stress": 0.09},
        ],
    },
    "ecommerce": {
        "label": "E-commerce",
        "data": [
            {"ticker": "SHOP",  "mci": 72, "divergence": -0.05, "signal": "Neutral",  "nsi_sigma":  0.2, "qa_stress": 0.10},
            {"ticker": "EBAY",  "mci": 66, "divergence": -0.09, "signal": "Neutral",  "nsi_sigma":  0.3, "qa_stress": 0.12},
            {"ticker": "ETSY",  "mci": 61, "divergence": -0.15, "signal": "Watch",    "nsi_sigma":  0.6, "qa_stress": 0.15},
            {"ticker": "W",     "mci": 53, "divergence": -0.24, "signal": "Watch",    "nsi_sigma":  1.0, "qa_stress": 0.22},
            {"ticker": "BABA",  "mci": 59, "divergence": -0.18, "signal": "Watch",    "nsi_sigma":  0.7, "qa_stress": 0.18},
            {"ticker": "JD",    "mci": 57, "divergence": -0.16, "signal": "Watch",    "nsi_sigma":  0.6, "qa_stress": 0.17},
        ],
    },
    "banks": {
        "label": "Major Banks",
        "data": [
            {"ticker": "JPM",   "mci": 79, "divergence":  0.10, "signal": "Positive", "nsi_sigma": -0.3, "qa_stress": 0.06},
            {"ticker": "BAC",   "mci": 71, "divergence": -0.06, "signal": "Neutral",  "nsi_sigma":  0.1, "qa_stress": 0.10},
            {"ticker": "WFC",   "mci": 67, "divergence": -0.10, "signal": "Neutral",  "nsi_sigma":  0.3, "qa_stress": 0.12},
            {"ticker": "C",     "mci": 63, "divergence": -0.13, "signal": "Neutral",  "nsi_sigma":  0.4, "qa_stress": 0.14},
            {"ticker": "GS",    "mci": 75, "divergence":  0.04, "signal": "Positive", "nsi_sigma": -0.2, "qa_stress": 0.08},
            {"ticker": "MS",    "mci": 73, "divergence": -0.02, "signal": "Neutral",  "nsi_sigma":  0.0, "qa_stress": 0.09},
            {"ticker": "USB",   "mci": 68, "divergence": -0.07, "signal": "Neutral",  "nsi_sigma":  0.2, "qa_stress": 0.11},
        ],
    },
    "fintech_payments": {
        "label": "Fintech & Payments",
        "data": [
            {"ticker": "V",     "mci": 83, "divergence":  0.16, "signal": "Positive", "nsi_sigma": -0.4, "qa_stress": 0.04},
            {"ticker": "MA",    "mci": 81, "divergence":  0.13, "signal": "Positive", "nsi_sigma": -0.3, "qa_stress": 0.05},
            {"ticker": "PYPL",  "mci": 62, "divergence": -0.15, "signal": "Neutral",  "nsi_sigma":  0.5, "qa_stress": 0.14},
            {"ticker": "SQ",    "mci": 66, "divergence": -0.10, "signal": "Neutral",  "nsi_sigma":  0.3, "qa_stress": 0.12},
            {"ticker": "AFRM",  "mci": 54, "divergence": -0.22, "signal": "Watch",    "nsi_sigma":  0.9, "qa_stress": 0.20},
            {"ticker": "SOFI",  "mci": 58, "divergence": -0.18, "signal": "Watch",    "nsi_sigma":  0.7, "qa_stress": 0.17},
            {"ticker": "NU",    "mci": 70, "divergence": -0.04, "signal": "Neutral",  "nsi_sigma":  0.1, "qa_stress": 0.09},
        ],
    },
    "pharma": {
        "label": "Pharmaceuticals",
        "data": [
            {"ticker": "JNJ",   "mci": 76, "divergence":  0.06, "signal": "Positive", "nsi_sigma": -0.2, "qa_stress": 0.07},
            {"ticker": "PFE",   "mci": 63, "divergence": -0.14, "signal": "Neutral",  "nsi_sigma":  0.5, "qa_stress": 0.13},
            {"ticker": "ABBV",  "mci": 74, "divergence": -0.02, "signal": "Neutral",  "nsi_sigma":  0.1, "qa_stress": 0.09},
            {"ticker": "MRK",   "mci": 77, "divergence":  0.09, "signal": "Positive", "nsi_sigma": -0.3, "qa_stress": 0.06},
            {"ticker": "LLY",   "mci": 85, "divergence":  0.18, "signal": "Positive", "nsi_sigma": -0.6, "qa_stress": 0.04},
            {"ticker": "BMY",   "mci": 67, "divergence": -0.08, "signal": "Neutral",  "nsi_sigma":  0.3, "qa_stress": 0.11},
            {"ticker": "AZN",   "mci": 72, "divergence": -0.04, "signal": "Neutral",  "nsi_sigma":  0.1, "qa_stress": 0.09},
        ],
    },
    "biotech": {
        "label": "Biotech",
        "data": [
            {"ticker": "MRNA",  "mci": 60, "divergence": -0.17, "signal": "Watch",    "nsi_sigma":  0.8, "qa_stress": 0.16},
            {"ticker": "REGN",  "mci": 78, "divergence":  0.09, "signal": "Positive", "nsi_sigma": -0.3, "qa_stress": 0.07},
            {"ticker": "BIIB",  "mci": 64, "divergence": -0.12, "signal": "Neutral",  "nsi_sigma":  0.4, "qa_stress": 0.13},
            {"ticker": "GILD",  "mci": 68, "divergence": -0.07, "signal": "Neutral",  "nsi_sigma":  0.2, "qa_stress": 0.10},
            {"ticker": "AMGN",  "mci": 73, "divergence":  0.02, "signal": "Neutral",  "nsi_sigma":  0.0, "qa_stress": 0.09},
            {"ticker": "VRTX",  "mci": 80, "divergence":  0.11, "signal": "Positive", "nsi_sigma": -0.4, "qa_stress": 0.06},
        ],
    },
    "healthtech": {
        "label": "Healthcare Tech",
        "data": [
            {"ticker": "ISRG",  "mci": 82, "divergence":  0.14, "signal": "Positive", "nsi_sigma": -0.5, "qa_stress": 0.05},
            {"ticker": "DXCM",  "mci": 71, "divergence": -0.06, "signal": "Neutral",  "nsi_sigma":  0.2, "qa_stress": 0.10},
            {"ticker": "TDOC",  "mci": 53, "divergence": -0.24, "signal": "Watch",    "nsi_sigma":  1.0, "qa_stress": 0.23},
            {"ticker": "VEEV",  "mci": 77, "divergence":  0.07, "signal": "Positive", "nsi_sigma": -0.2, "qa_stress": 0.07},
            {"ticker": "IDXX",  "mci": 75, "divergence":  0.04, "signal": "Positive", "nsi_sigma": -0.1, "qa_stress": 0.07},
            {"ticker": "ILMN",  "mci": 62, "divergence": -0.14, "signal": "Neutral",  "nsi_sigma":  0.5, "qa_stress": 0.14},
        ],
    },
    "energy": {
        "label": "Energy",
        "data": [
            {"ticker": "XOM",   "mci": 74, "divergence": -0.03, "signal": "Neutral",  "nsi_sigma":  0.1, "qa_stress": 0.09},
            {"ticker": "CVX",   "mci": 76, "divergence":  0.05, "signal": "Positive", "nsi_sigma": -0.2, "qa_stress": 0.07},
            {"ticker": "COP",   "mci": 71, "divergence": -0.07, "signal": "Neutral",  "nsi_sigma":  0.2, "qa_stress": 0.10},
            {"ticker": "SLB",   "mci": 68, "divergence": -0.09, "signal": "Neutral",  "nsi_sigma":  0.3, "qa_stress": 0.11},
            {"ticker": "EOG",   "mci": 72, "divergence": -0.04, "signal": "Neutral",  "nsi_sigma":  0.1, "qa_stress": 0.09},
            {"ticker": "PSX",   "mci": 69, "divergence": -0.06, "signal": "Neutral",  "nsi_sigma":  0.2, "qa_stress": 0.10},
            {"ticker": "VLO",   "mci": 67, "divergence": -0.08, "signal": "Neutral",  "nsi_sigma":  0.3, "qa_stress": 0.11},
        ],
    },
    "consumer_staples": {
        "label": "Consumer Staples",
        "data": [
            {"ticker": "PG",    "mci": 78, "divergence":  0.08, "signal": "Positive", "nsi_sigma": -0.3, "qa_stress": 0.06},
            {"ticker": "KO",    "mci": 76, "divergence":  0.06, "signal": "Positive", "nsi_sigma": -0.2, "qa_stress": 0.07},
            {"ticker": "PEP",   "mci": 74, "divergence":  0.03, "signal": "Neutral",  "nsi_sigma":  0.0, "qa_stress": 0.08},
            {"ticker": "WMT",   "mci": 77, "divergence":  0.07, "signal": "Positive", "nsi_sigma": -0.3, "qa_stress": 0.06},
            {"ticker": "COST",  "mci": 80, "divergence":  0.12, "signal": "Positive", "nsi_sigma": -0.4, "qa_stress": 0.05},
            {"ticker": "CL",    "mci": 71, "divergence": -0.05, "signal": "Neutral",  "nsi_sigma":  0.1, "qa_stress": 0.09},
            {"ticker": "MO",    "mci": 65, "divergence": -0.11, "signal": "Neutral",  "nsi_sigma":  0.3, "qa_stress": 0.12},
        ],
    },
    "consumer_discretionary": {
        "label": "Consumer Discretionary",
        "data": [
            {"ticker": "NKE",   "mci": 73, "divergence": -0.04, "signal": "Neutral",  "nsi_sigma":  0.1, "qa_stress": 0.09},
            {"ticker": "MCD",   "mci": 78, "divergence":  0.08, "signal": "Positive", "nsi_sigma": -0.3, "qa_stress": 0.06},
            {"ticker": "SBUX",  "mci": 65, "divergence": -0.12, "signal": "Neutral",  "nsi_sigma":  0.5, "qa_stress": 0.14},
            {"ticker": "HD",    "mci": 76, "divergence":  0.05, "signal": "Positive", "nsi_sigma": -0.2, "qa_stress": 0.07},
            {"ticker": "TGT",   "mci": 64, "divergence": -0.13, "signal": "Neutral",  "nsi_sigma":  0.4, "qa_stress": 0.13},
            {"ticker": "LOW",   "mci": 72, "divergence": -0.03, "signal": "Neutral",  "nsi_sigma":  0.1, "qa_stress": 0.09},
            {"ticker": "BKNG",  "mci": 75, "divergence":  0.04, "signal": "Positive", "nsi_sigma": -0.2, "qa_stress": 0.07},
        ],
    },
    "industrial": {
        "label": "Industrial",
        "data": [
            {"ticker": "CAT",   "mci": 77, "divergence":  0.07, "signal": "Positive", "nsi_sigma": -0.3, "qa_stress": 0.06},
            {"ticker": "HON",   "mci": 74, "divergence":  0.02, "signal": "Neutral",  "nsi_sigma":  0.0, "qa_stress": 0.08},
            {"ticker": "GE",    "mci": 71, "divergence": -0.06, "signal": "Neutral",  "nsi_sigma":  0.2, "qa_stress": 0.10},
            {"ticker": "MMM",   "mci": 60, "divergence": -0.17, "signal": "Watch",    "nsi_sigma":  0.7, "qa_stress": 0.17},
            {"ticker": "EMR",   "mci": 73, "divergence":  0.01, "signal": "Neutral",  "nsi_sigma":  0.0, "qa_stress": 0.08},
            {"ticker": "ETN",   "mci": 76, "divergence":  0.06, "signal": "Positive", "nsi_sigma": -0.2, "qa_stress": 0.07},
            {"ticker": "PH",    "mci": 74, "divergence":  0.03, "signal": "Neutral",  "nsi_sigma":  0.0, "qa_stress": 0.08},
        ],
    },
    "aerospace_defense": {
        "label": "Aerospace & Defense",
        "data": [
            {"ticker": "BA",    "mci": 56, "divergence": -0.22, "signal": "Watch",    "nsi_sigma":  0.9, "qa_stress": 0.21},
            {"ticker": "LMT",   "mci": 76, "divergence":  0.05, "signal": "Positive", "nsi_sigma": -0.2, "qa_stress": 0.07},
            {"ticker": "RTX",   "mci": 73, "divergence": -0.03, "signal": "Neutral",  "nsi_sigma":  0.1, "qa_stress": 0.09},
            {"ticker": "NOC",   "mci": 75, "divergence":  0.04, "signal": "Positive", "nsi_sigma": -0.1, "qa_stress": 0.08},
            {"ticker": "GD",    "mci": 74, "divergence":  0.02, "signal": "Neutral",  "nsi_sigma":  0.0, "qa_stress": 0.08},
            {"ticker": "HII",   "mci": 70, "divergence": -0.06, "signal": "Neutral",  "nsi_sigma":  0.2, "qa_stress": 0.10},
        ],
    },
    "telecom": {
        "label": "Telecom",
        "data": [
            {"ticker": "T",     "mci": 62, "divergence": -0.15, "signal": "Neutral",  "nsi_sigma":  0.4, "qa_stress": 0.14},
            {"ticker": "VZ",    "mci": 60, "divergence": -0.17, "signal": "Watch",    "nsi_sigma":  0.5, "qa_stress": 0.16},
            {"ticker": "TMUS",  "mci": 74, "divergence":  0.02, "signal": "Neutral",  "nsi_sigma":  0.0, "qa_stress": 0.08},
            {"ticker": "CHTR",  "mci": 65, "divergence": -0.11, "signal": "Neutral",  "nsi_sigma":  0.3, "qa_stress": 0.12},
            {"ticker": "LUMN",  "mci": 44, "divergence": -0.35, "signal": "Watch",    "nsi_sigma":  1.5, "qa_stress": 0.28},
            {"ticker": "CMCSA", "mci": 67, "divergence": -0.08, "signal": "Neutral",  "nsi_sigma":  0.2, "qa_stress": 0.11},
        ],
    },
    "ai_data": {
        "label": "AI & Data Analytics",
        "data": [
            {"ticker": "PLTR",  "mci": 72, "divergence": -0.05, "signal": "Neutral",  "nsi_sigma":  0.1, "qa_stress": 0.10},
            {"ticker": "IBM",   "mci": 68, "divergence": -0.08, "signal": "Neutral",  "nsi_sigma":  0.2, "qa_stress": 0.11},
            {"ticker": "AI",    "mci": 59, "divergence": -0.19, "signal": "Watch",    "nsi_sigma":  0.8, "qa_stress": 0.18},
            {"ticker": "PATH",  "mci": 63, "divergence": -0.13, "signal": "Neutral",  "nsi_sigma":  0.4, "qa_stress": 0.13},
            {"ticker": "SMAR",  "mci": 67, "divergence": -0.09, "signal": "Neutral",  "nsi_sigma":  0.2, "qa_stress": 0.11},
            {"ticker": "BBAI",  "mci": 51, "divergence": -0.27, "signal": "Watch",    "nsi_sigma":  1.0, "qa_stress": 0.24},
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


def get_mock_peer_data(ticker, live_mci=None):
    """
    Return (df, sector_label) for the ticker's sector peer group.
    Adds rank and delta_vs_peer_avg columns.
    Live MCI replaces illustrative value for the selected ticker.
    """
    ticker_upper = ticker.upper()
    sector_key, sector_label = _get_sector(ticker_upper)
    group = _PEER_UNIVERSE[sector_key]

    rows = [
        {**item, "is_selected": item["ticker"] == ticker_upper}
        for item in group["data"]
    ]
    df = pd.DataFrame(rows)

    if ticker_upper not in df["ticker"].values:
        new_row = {
            "ticker": ticker_upper,
            "mci": round(live_mci, 1) if live_mci else 50,
            "divergence": 0.0,
            "signal": "Live",
            "is_selected": True,
        }
        df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
    elif live_mci is not None:
        df.loc[df["ticker"] == ticker_upper, "mci"]    = round(live_mci, 1)
        df.loc[df["ticker"] == ticker_upper, "signal"] = "Live"

    df = df.sort_values("mci", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    peer_avg = df.loc[~df["is_selected"], "mci"].mean()
    df["delta_vs_peers"] = (df["mci"] - peer_avg).round(1)

    return df, sector_label


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
        **dict(zip(("peer_data","peer_sector"), get_mock_peer_data(ticker, multimodal_result.get("mci")))),
    }
