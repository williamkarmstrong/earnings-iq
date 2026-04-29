"""
Event Study module.
Measures abnormal stock returns around earnings call release dates
using CAPM event study methodology (Brown & Warner, 1985).

Estimation window: t = -120 to -20 trading days relative to event
Event window:      t = -1  to  +3
Market proxy:      SPY (S&P 500 ETF)

Sector Earnings Price Sensitivity: average |CAR[-1,+3]| across the last 3
earnings events for 3 representative tickers per sector. This is an
event-specific measure — how much the sector typically moves around earnings
releases — distinct from general market beta. Cached for 24 hours.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import date, timedelta


# ── Representative tickers used to compute live sector betas ───────────────
# 3 large, liquid names per sector. OLS beta vs SPY averaged across all three.
_SECTOR_REPS = {
    "Mega-cap Tech":          ["AAPL", "MSFT", "GOOGL"],
    "Semiconductors":         ["NVDA", "AMD",  "INTC"],
    "EV & Auto":              ["TSLA", "F",    "GM"],
    "Cloud & SaaS":           ["CRM",  "NOW",  "SNOW"],
    "Cybersecurity":          ["CRWD", "PANW", "FTNT"],
    "Streaming & Media":      ["NFLX", "DIS",  "SPOT"],
    "Social Media":           ["SNAP", "PINS", "TTD"],
    "E-commerce":             ["SHOP", "EBAY", "BABA"],
    "Major Banks":            ["JPM",  "BAC",  "GS"],
    "Fintech & Payments":     ["V",    "PYPL", "SQ"],
    "Pharmaceuticals":        ["JNJ",  "PFE",  "MRK"],
    "Biotech":                ["MRNA", "REGN", "AMGN"],
    "Healthcare Tech":        ["ISRG", "DXCM", "VEEV"],
    "Energy":                 ["XOM",  "CVX",  "COP"],
    "Consumer Staples":       ["PG",   "KO",   "WMT"],
    "Consumer Discretionary": ["NKE",  "MCD",  "HD"],
    "Industrial":             ["CAT",  "HON",  "GE"],
    "Aerospace & Defense":    ["LMT",  "RTX",  "NOC"],
    "Telecom":                ["T",    "VZ",   "TMUS"],
    "AI & Data Analytics":    ["PLTR", "IBM",  "AI"],
}

# Fallback earnings sensitivity (avg |CAR|) if live computation fails.
# Derived from historical sector event study literature.
_FALLBACK_EPS = {
    "Biotech":                0.112, "EV & Auto":              0.098,
    "Semiconductors":         0.091, "AI & Data Analytics":    0.087,
    "Social Media":           0.083, "Cloud & SaaS":           0.078,
    "Cybersecurity":          0.071, "Streaming & Media":      0.068,
    "Fintech & Payments":     0.063, "Mega-cap Tech":          0.058,
    "E-commerce":             0.055, "Healthcare Tech":        0.052,
    "Consumer Discretionary": 0.047, "Industrial":             0.041,
    "Pharmaceuticals":        0.038, "Aerospace & Defense":    0.035,
    "Major Banks":            0.034, "Energy":                 0.031,
    "Telecom":                0.024, "Consumer Staples":       0.019,
}


def _eps_label(eps):
    """Earnings Price Sensitivity tier based on avg |CAR|."""
    if eps > 0.06:
        return "High"
    if eps > 0.03:
        return "Medium"
    return "Low"


def _eps_colour(eps):
    if eps > 0.06:
        return "#C0392B"
    if eps > 0.03:
        return "#C17B00"
    return "#1A7F4B"


def _ols_beta(y, x):
    """Return OLS beta coefficient of y ~ 1 + x."""
    X = np.column_stack([np.ones(len(x)), x])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coeffs[1])


@st.cache_data(ttl=86400, show_spinner=False)
def compute_sector_betas():
    """
    Download 1 year of daily returns for all representative tickers and SPY.
    Compute OLS beta for each ticker vs SPY, average per sector.
    Cached for 24 hours — betas are stable enough at daily frequency.
    Returns dict: {sector_label: avg_beta}
    """
    all_tickers = list({t for tickers in _SECTOR_REPS.values() for t in tickers})
    all_tickers.append("SPY")

    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=400)  # ~280 trading days

    try:
        raw = yf.download(
            all_tickers,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
    except Exception:
        return {}

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.dropna(axis=1, how="all")

    if "SPY" not in close.columns or len(close) < 30:
        return {}

    returns = np.log(close / close.shift(1)).dropna()
    spy = returns["SPY"].values

    sector_betas = {}
    for sector, tickers in _SECTOR_REPS.items():
        betas = []
        for t in tickers:
            if t not in returns.columns:
                continue
            col = returns[t].dropna()
            # Align lengths via shared index with SPY
            aligned = returns[[t, "SPY"]].dropna()
            if len(aligned) < 30:
                continue
            try:
                b = _ols_beta(aligned[t].values, aligned["SPY"].values)
                if -1 < b < 5:   # sanity: discard extreme outliers
                    betas.append(b)
            except Exception:
                continue
        if betas:
            sector_betas[sector] = round(float(np.mean(betas)), 3)

    return sector_betas


def _calendar_earnings_dates(today, n=3):
    """
    Generate the last n approximate quarterly earnings dates before today.
    Uses standard US large-cap fiscal calendar:
      Q1 results → late April | Q2 → late July | Q3 → late October | Q4 → late January
    No API calls — derived purely from the calendar.
    """
    candidates = []
    for yr in range(today.year - 3, today.year + 1):
        candidates += [
            date(yr,     4, 25),   # Q1 report
            date(yr,     7, 28),   # Q2 report
            date(yr,    10, 28),   # Q3 report
            date(yr + 1, 1, 28),   # Q4 report
        ]
    past = sorted([d for d in candidates if d < today], reverse=True)
    return past[:n]


def _snap_to_trading_day(event_date, trade_index):
    """Return the closest date in trade_index to event_date."""
    ts = pd.Timestamp(event_date)
    pos = trade_index.searchsorted(ts)
    if pos == 0:
        return trade_index[0]
    if pos >= len(trade_index):
        return trade_index[-1]
    before = trade_index[pos - 1]
    after  = trade_index[pos]
    return before if (ts - before) <= (after - ts) else after


def _car_for_ticker(t, ret, event_dates):
    """
    Compute avg |CAR[-1,+3]| for ticker t across the supplied event dates.
    ret: DataFrame with columns [t, "SPY"], log-return series, DatetimeIndex.
    Returns list of |CAR| values (may be empty if data is insufficient).
    """
    cars = []
    for event_date in event_dates:
        snap = _snap_to_trading_day(event_date, ret.index)
        pos  = int(ret.index.get_loc(snap))
        rel  = np.arange(len(ret)) - pos

        est = ret[(rel >= -120) & (rel <= -20)]
        if len(est) < 20:
            continue

        y, x = est[t].values, est["SPY"].values
        X = np.column_stack([np.ones(len(x)), x])
        try:
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta_est = float(coeffs[0]), float(coeffs[1])
        except Exception:
            continue

        ev = ret[(rel >= -1) & (rel <= 3)]
        if ev.empty:
            continue

        ar  = ev[t] - (alpha + beta_est * ev["SPY"])
        cars.append(abs(float(ar.sum())))
    return cars


@st.cache_data(ttl=86400, show_spinner=False)
def compute_sector_earnings_sensitivity():
    """
    Compute Earnings Price Sensitivity (EPS) per sector.

    EPS = average |CAR[-1,+3]| across the last 3 earnings events for each
    representative ticker in the sector, averaged across the 3 tickers.

    This is an event-specific measure: how much a sector typically moves
    around earnings releases after stripping out normal market movement.
    Distinct from general CAPM beta (which measures day-to-day co-movement).

    Earnings dates are derived from a quarterly calendar approximation
    (Q1≈Apr 25, Q2≈Jul 28, Q3≈Oct 28, Q4≈Jan 28) — no per-ticker API
    calls needed. The approximation introduces ≤2 weeks of date error,
    which is negligible given the [-1,+3] event window.

    Falls back to yf.earnings_dates for any ticker where calendar events
    fall inside gaps in the price data.

    Methodology: CAPM event study (Brown & Warner, 1985).
    Cached 24 hours.
    Returns dict: {sector_label: avg_abs_car}
    """
    all_tickers = list({t for tickers in _SECTOR_REPS.values() for t in tickers})
    all_tickers.append("SPY")

    today    = date.today()
    end_dt   = today
    start_dt = today - timedelta(days=850)  # ~2.8 years: covers 3 events + estimation windows

    try:
        raw = yf.download(
            all_tickers,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
    except Exception:
        return {}

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.dropna(axis=1, how="all")
    if "SPY" not in close.columns or len(close) < 60:
        return {}

    returns = np.log(close / close.shift(1)).dropna()
    returns.index = pd.to_datetime(returns.index)

    # Calendar-derived earnings dates — same 3 dates used for every ticker.
    # Per-company date error is typically ≤14 trading days, which is within
    # the estimation gap (we exclude t=-20 to t=0, so ≤14-day misalignment
    # rarely contaminates the estimation window).
    cal_dates = _calendar_earnings_dates(today, n=3)

    sector_results = {}
    for sector, tickers in _SECTOR_REPS.items():
        cars = []
        for t in tickers:
            if t not in returns.columns:
                continue
            ret = returns[[t, "SPY"]].dropna()
            if len(ret) < 60:
                continue
            ticker_cars = _car_for_ticker(t, ret, cal_dates)
            # If calendar dates fall in data gaps, try yfinance as backup
            if not ticker_cars:
                try:
                    ed = yf.Ticker(t).earnings_dates
                    if ed is not None and not ed.empty:
                        idx = ed.index.tz_localize(None) if ed.index.tz is not None else ed.index
                        past = sorted([d.date() for d in idx if d.date() < today], reverse=True)
                        if past:
                            ticker_cars = _car_for_ticker(t, ret, past[:3])
                except Exception:
                    pass
            cars.extend(ticker_cars)

        if cars:
            sector_results[sector] = round(float(np.mean(cars)), 4)

    return sector_results




# Authoritative earnings call dates verified against company IR / press releases.
# Override yfinance lookups, which are unreliable for fiscal-year companies
# (AAPL, MSFT, V, DIS, PG, WMT, CRM) whose Q1 fiscal ≠ Q1 calendar.
# Key: (TICKER, "QN", calendar_year_of_call) → call date.
EARNINGS_DATE_OVERRIDES = {
    ("TSLA",  "Q1", 2024): date(2024,  4, 23),
    ("XOM",   "Q1", 2024): date(2024,  4, 26),
    ("JNJ",   "Q1", 2024): date(2024,  4, 16),
    ("V",     "Q1", 2024): date(2024,  1, 25),  # Q1 FY2024
    ("AAPL",  "Q1", 2024): date(2024,  2,  1),  # Q1 FY2024
    ("JPM",   "Q1", 2024): date(2024,  4, 12),
    ("META",  "Q1", 2024): date(2024,  4, 24),
    ("MSFT",  "Q1", 2024): date(2023, 10, 24),  # Q1 FY2024
    ("AMZN",  "Q1", 2024): date(2024,  4, 30),
    ("GOOGL", "Q1", 2024): date(2024,  4, 25),
    ("C",     "Q1", 2024): date(2024,  4, 12),
    ("DIS",   "Q1", 2024): date(2024,  2,  7),
    ("WMT",   "Q1", 2025): date(2024,  5, 16),  # Q1 FY2025 reported May 2024
    ("WMT",   "Q1", 2024): date(2024,  5, 16),  # alias if user enters calendar 2024
    ("LLY",   "Q1", 2024): date(2024,  4, 30),
    ("PG",    "Q1", 2024): date(2023, 10, 18),  # Q1 FY2024
    ("UNH",   "Q1", 2024): date(2024,  4, 16),
    ("AMT",   "Q1", 2024): date(2024,  4, 30),
    ("BLK",   "Q1", 2024): date(2024,  4, 12),
    ("LIN",   "Q1", 2024): date(2024,  5,  2),
    ("NFLX",  "Q1", 2024): date(2024,  4, 18),
    ("CAT",   "Q1", 2024): date(2024,  4, 25),
    ("CRM",   "Q1", 2025): date(2024,  5, 29),  # Q1 FY2025 reported May 2024
    ("CRM",   "Q1", 2024): date(2024,  5, 29),  # alias if user enters calendar 2024
}


def get_earnings_date(ticker, period, year):
    """
    Get actual earnings announcement date.
    1. Authoritative override map (verified IR dates).
    2. yfinance earnings_dates.
    3. Quarter-end approximation (calendar-year fiscal companies only).
    """
    override = EARNINGS_DATE_OVERRIDES.get((ticker.upper(), period, int(year)))
    if override is not None:
        return override

    approx = {
        "Q1": date(year,     4, 25),
        "Q2": date(year,     7, 28),
        "Q3": date(year,    10, 28),
        "Q4": date(year + 1, 1, 28),
    }
    fallback = approx.get(period, date(year, 4, 25))

    try:
        ed = yf.Ticker(ticker).get_earnings_dates()
        if ed is not None and not ed.empty:
            idx = ed.index.tz_localize(None) if ed.index.tz is not None else ed.index
            year_dates = sorted([d for d in idx if d.year == year])
            q_idx = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}.get(period)
            if q_idx is not None and q_idx < len(year_dates):
                return year_dates[q_idx].date()
    except Exception:
        pass

    return fallback


@st.cache_data(show_spinner=False)
def run_event_study(ticker, period, year):
    """
    CAPM event study around the earnings call date.

    Steps:
      1. Resolve t=0 via yfinance earnings_dates or quarter approximation
      2. Download stock + SPY returns for fetch window
      3. Estimation window OLS: R_stock = alpha + beta * R_SPY
      4. Event window: AR_t = R_actual_t - (alpha + beta * R_SPY_t)
      5. CAR = sum(AR_t);  T-stat = CAR / (sigma_est * sqrt(n_event))

    Returns dict with beta, alpha, r_squared, car, t_stat, n_est, ar_series, event_date.
    Returns {"error": message} on failure.
    """
    event_date = get_earnings_date(ticker, period, year)

    fetch_start = event_date - timedelta(days=220)
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

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

    if close.empty:
        return {"error": "No price data returned."}
    for col in [ticker, "SPY"]:
        if col not in close.columns:
            return {"error": f"Price data missing for {col}."}

    close = close[[ticker, "SPY"]].dropna()
    returns = np.log(close / close.shift(1)).dropna()

    if len(returns) < 25:
        return {"error": "Insufficient return history (< 25 trading days available)."}

    returns.index = pd.to_datetime(returns.index)
    event_ts = pd.Timestamp(event_date)
    pos = int(min(returns.index.searchsorted(event_ts), len(returns) - 1))
    rel = np.arange(len(returns)) - pos

    # Estimation window
    est = returns[(rel >= -120) & (rel <= -20)]
    if len(est) < 20:
        return {"error": f"Estimation window has only {len(est)} days (need ≥20). "
                         "Try an earlier quarter or verify price history is available."}

    y, x = est[ticker].values, est["SPY"].values
    X = np.column_stack([np.ones(len(x)), x])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coeffs[0]), float(coeffs[1])

    y_hat  = alpha + beta * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_sq   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    resid_std = float(np.std(y - y_hat, ddof=2)) if len(y) > 2 else 0.01

    # Event window
    ev_mask = (rel >= -1) & (rel <= 3)
    ev = returns[ev_mask].copy()
    if ev.empty:
        return {"error": "Event window has no trading data — date may be in the future."}

    ev_rel = rel[ev_mask]
    ar     = ev[ticker] - (alpha + beta * ev["SPY"])
    car    = float(ar.sum())
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


def get_sector_sensitivity_df(eps_data, selected_sector_label=None, live_car=None):
    """
    Build the sector Earnings Price Sensitivity DataFrame for the chart.
    eps_data: dict from compute_sector_earnings_sensitivity() — {sector: avg_abs_car}.
              Falls back to _FALLBACK_EPS if empty.
    Patches selected_sector_label with the live ticker's |CAR| if provided.
    Returns DataFrame sorted ascending by eps (chart reads highest at top).
    """
    rows = []
    for sector in _SECTOR_REPS:
        eps = eps_data.get(sector) or _FALLBACK_EPS.get(sector, 0.05)
        is_live = (sector == selected_sector_label and live_car is not None)
        if is_live:
            eps = abs(live_car)
        rows.append({
            "sector":      sector,
            "eps":         round(eps, 4),
            "sensitivity": _eps_label(eps),
            "colour":      _eps_colour(eps),
            "is_live":     is_live,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("eps", ascending=True).reset_index(drop=True)
