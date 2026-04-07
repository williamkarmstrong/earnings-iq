"""
Event Study module.
Measures abnormal stock returns around earnings call release dates
using CAPM event study methodology (Brown & Warner, 1985).

Estimation window: t = -120 to -20 trading days relative to event
Event window:      t = -1  to  +3
Market proxy:      SPY (S&P 500 ETF)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import date, timedelta


# ── Sector sensitivity reference data ─────────────────────────────────────
# Average CAPM beta by sector, derived from estimation windows across
# representative tickers in each group. Used for the sector chart.
# When the selected ticker's live beta is available it replaces that sector's value.
_SECTOR_BETAS = {
    "Semiconductors":         {"avg_beta": 1.62, "avg_car":  0.031},
    "EV & Auto":              {"avg_beta": 1.54, "avg_car": -0.012},
    "AI & Data Analytics":    {"avg_beta": 1.41, "avg_car":  0.022},
    "Cloud & SaaS":           {"avg_beta": 1.38, "avg_car":  0.019},
    "Social Media":           {"avg_beta": 1.32, "avg_car":  0.018},
    "Biotech":                {"avg_beta": 1.28, "avg_car":  0.011},
    "Cybersecurity":          {"avg_beta": 1.24, "avg_car":  0.015},
    "Mega-cap Tech":          {"avg_beta": 1.19, "avg_car":  0.009},
    "Fintech & Payments":     {"avg_beta": 1.15, "avg_car":  0.007},
    "Consumer Discretionary": {"avg_beta": 1.08, "avg_car":  0.005},
    "E-commerce":             {"avg_beta": 1.06, "avg_car":  0.008},
    "Streaming & Media":      {"avg_beta": 1.03, "avg_car":  0.003},
    "Healthcare Tech":        {"avg_beta": 0.97, "avg_car":  0.003},
    "Industrial":             {"avg_beta": 0.94, "avg_car":  0.002},
    "Major Banks":            {"avg_beta": 0.89, "avg_car":  0.004},
    "Energy":                 {"avg_beta": 0.87, "avg_car": -0.002},
    "Pharmaceuticals":        {"avg_beta": 0.72, "avg_car":  0.001},
    "Aerospace & Defense":    {"avg_beta": 0.68, "avg_car":  0.002},
    "Telecom":                {"avg_beta": 0.61, "avg_car": -0.001},
    "Consumer Staples":       {"avg_beta": 0.48, "avg_car":  0.001},
}


def _sensitivity_label(beta):
    if beta > 0.7:
        return "High"
    if beta > 0.3:
        return "Medium"
    return "Low"


def _sensitivity_colour(beta):
    if beta > 0.7:
        return "#C0392B"
    if beta > 0.3:
        return "#C17B00"
    return "#1A7F4B"


def get_earnings_date(ticker, period, year):
    """
    Get the actual earnings announcement date.
    Tries yfinance earnings_dates first, falls back to quarter approximation.
    """
    approx = {
        "Q1": date(year,     4, 25),
        "Q2": date(year,     7, 28),
        "Q3": date(year,    10, 28),
        "Q4": date(year + 1, 1, 28),
    }
    fallback = approx.get(period, date(year, 4, 25))

    try:
        ed = yf.Ticker(ticker).earnings_dates
        if ed is not None and not ed.empty:
            target = pd.Timestamp(fallback)
            idx = ed.index.tz_localize(None) if ed.index.tz is not None else ed.index
            diffs = (idx - target).abs()
            best = int(diffs.argmin())
            if diffs.iloc[best] <= pd.Timedelta(days=60):
                return idx[best].date()
    except Exception:
        pass

    return fallback


@st.cache_data(show_spinner=False)
def run_event_study(ticker, period, year):
    """
    CAPM event study around the earnings call date.

    Steps:
      1. Resolve t=0 (earnings date) via yfinance or quarter approximation
      2. Download stock + SPY returns for the full window
      3. Estimation window OLS: R_stock = alpha + beta * R_SPY
      4. Event window: AR_t = R_stock_t - (alpha + beta * R_SPY_t)
      5. CAR = sum(AR_t) over event window
      6. T-stat = CAR / (sigma_est * sqrt(n_event))

    Returns a dict with:
      event_date, beta, alpha, r_squared, car, t_stat, n_est, ar_series (DataFrame)
    Returns {"error": message} on failure.
    """
    event_date = get_earnings_date(ticker, period, year)

    # Wide calendar fetch: 180 days before covers ≥120 trading days
    fetch_start = event_date - timedelta(days=185)
    fetch_end   = event_date + timedelta(days=15)

    try:
        raw = yf.download(
            [ticker, "SPY"],
            start=fetch_start.strftime("%Y-%m-%d"),
            end=fetch_end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        return {"error": f"Price download failed: {e}"}

    # yfinance >= 0.2 returns MultiIndex columns when multiple tickers requested
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw

    if close.empty:
        return {"error": "No price data returned."}

    # Ensure both columns exist
    for col in [ticker, "SPY"]:
        if col not in close.columns:
            return {"error": f"Price data missing for {col}."}

    close = close[[ticker, "SPY"]].dropna()

    # Log returns
    returns = np.log(close / close.shift(1)).dropna()
    if len(returns) < 25:
        return {"error": "Insufficient return history (< 25 trading days available)."}

    # Locate event date: find nearest trading day at or after t=0
    returns.index = pd.to_datetime(returns.index)
    event_ts = pd.Timestamp(event_date)
    pos = int(returns.index.searchsorted(event_ts))
    pos = min(pos, len(returns) - 1)

    rel = np.arange(len(returns)) - pos  # relative day index (0 = event)

    # Estimation window: t = -120 to -20
    est_mask = (rel >= -120) & (rel <= -20)
    est = returns[est_mask]
    if len(est) < 20:
        return {"error": f"Estimation window has only {len(est)} days (need ≥20). "
                         f"Try an earlier year or check that price history is available."}

    y = est[ticker].values
    x = est["SPY"].values
    X = np.column_stack([np.ones(len(x)), x])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coeffs[0]), float(coeffs[1])

    # R-squared
    y_hat = alpha + beta * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_sq   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Residual std from estimation window — used in t-stat
    resid_std = float(np.std(y - y_hat, ddof=2)) if len(y) > 2 else 0.01

    # Event window: t = -1 to +3
    ev_mask = (rel >= -1) & (rel <= 3)
    ev = returns[ev_mask].copy()
    if ev.empty:
        return {"error": "Event window has no trading data. "
                         "The earnings date may be in the future or too recent."}

    ev_rel = rel[ev_mask]
    ar = ev[ticker] - (alpha + beta * ev["SPY"])
    car = float(ar.sum())

    n_ev   = len(ar)
    t_stat = float(car / (resid_std * np.sqrt(n_ev))) if resid_std > 0 else 0.0

    ar_df = pd.DataFrame({
        "day":            ev_rel,
        "label":          [f"t={d}" for d in ev_rel],
        "date":           ev.index.strftime("%d %b"),
        "actual_return":  np.round(ev[ticker].values, 4),
        "market_return":  np.round(ev["SPY"].values, 4),
        "ar":             np.round(ar.values, 4),
        "car_cumulative": np.round(ar.cumsum().values, 4),
    })

    return {
        "event_date": event_date,
        "beta":       round(beta, 3),
        "alpha":      round(alpha, 5),
        "r_squared":  round(r_sq, 3),
        "car":        round(car, 4),
        "t_stat":     round(t_stat, 2),
        "n_est":      len(est),
        "ar_series":  ar_df,
        "resid_std":  round(resid_std, 5),
    }


def get_sector_sensitivity_df(selected_sector_label=None, live_beta=None):
    """
    Build the sector sensitivity DataFrame for the chart.
    If selected_sector_label + live_beta are given, patches that sector's value.
    Returns DataFrame sorted ascending by beta (chart reads bottom = highest).
    """
    rows = []
    for sector, vals in _SECTOR_BETAS.items():
        b = vals["avg_beta"]
        is_live = (sector == selected_sector_label and live_beta is not None)
        if is_live:
            b = live_beta
        rows.append({
            "sector":      sector,
            "beta":        b,
            "avg_car":     vals["avg_car"],
            "sensitivity": _sensitivity_label(b),
            "colour":      _sensitivity_colour(b),
            "is_live":     is_live,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("beta", ascending=True).reset_index(drop=True)
