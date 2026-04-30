"""
Recompute and save the Narrative Shift Index for a demo-cached earnings call.

Usage:
    python scripts/recompute_nsi.py --ticker TSLA --period Q1 --year 2024

Note: run with the project's Python environment (miniforge/conda), e.g.:
    python scripts/recompute_nsi.py ...

Fetches/uses cached transcripts for the current and previous N quarters,
reruns compute_nsi(), and writes the updated value back into
demo/{ticker}_{year}_{period}/results.json.
"""
import argparse
import json
import os
import sys
import types

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Minimal streamlit stub so app modules (nlp, ingestion) can be imported
# without a running Streamlit server.
# ---------------------------------------------------------------------------
def _noop_decorator(*args, **kwargs):
    def wrapper(fn):
        return fn
    # Called as @st.cache_data or @st.cache_data(show_spinner=False)
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return wrapper

def _load_streamlit_secrets() -> dict:
    """Parse .streamlit/secrets.toml so the stub can serve API keys."""
    toml_path = os.path.join(_root, ".streamlit", "secrets.toml")
    if not os.path.exists(toml_path):
        return {}
    try:
        import tomllib  # Python 3.11+
        with open(toml_path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        result = {}
        with open(toml_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    result[k.strip()] = v.strip().strip('"').strip("'")
        return result

class _Secrets(dict):
    def __init__(self):
        super().__init__(_load_streamlit_secrets())

    def get(self, key, default=None):
        return self[key] if key in self else os.environ.get(key, default)

_st_stub = types.ModuleType("streamlit")
_st_stub.cache_data     = _noop_decorator
_st_stub.cache_resource = _noop_decorator
_st_stub.secrets        = _Secrets()
sys.modules["streamlit"] = _st_stub

# ---------------------------------------------------------------------------
# Now safe to import app modules
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(_root, "app"))
os.chdir(_root)  # ingestion resolves cache/ relative to cwd

from dotenv import load_dotenv
load_dotenv()

from ingestion import fetch_transcript
from nlp import analyse_transcript_text, compute_nsi
from utils import get_previous_quarters


def _transcript_to_stats(ticker: str, period: str, year: int, label: str) -> dict | None:
    av_json, err = fetch_transcript(ticker, period, year)
    if not av_json:
        print(f"  [{label}] no transcript — {err}")
        return None
    text = " ".join((item.get("content", "") or "") for item in av_json)
    stats = analyse_transcript_text(text)
    print(f"  [{label}] sentiment={stats['sentiment']:+.3f}  hedge={stats['hedge_freq']:.2f}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Recompute NSI for a demo-cached earnings call.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--period", required=True, choices=["Q1", "Q2", "Q3", "Q4"])
    parser.add_argument("--year",   required=True, type=int)
    parser.add_argument("--n-history", default=4, type=int,
                        help="Number of prior quarters to fetch (default: 4)")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    period = args.period
    year   = args.year

    demo_json = os.path.join("demo", f"{ticker}_{year}_{period}", "results.json")
    if not os.path.exists(demo_json):
        sys.exit(f"No demo cache found at {demo_json} — run the full pipeline first.")

    print(f"\nRecomputing NSI for {ticker} {period} {year}")
    print(f"Demo cache: {demo_json}\n")

    # Current quarter
    print("Current quarter:")
    current_stats = _transcript_to_stats(ticker, period, year, f"{period} {year} (current)")
    if current_stats is None:
        sys.exit("Cannot compute NSI without a transcript for the current quarter.")
    current_stats["label"] = f"{period} {year} (current)"

    # Previous quarters
    print(f"\nPrevious {args.n_history} quarters:")
    historical_stats = []
    qoq_data = [current_stats]
    for prev_period, prev_year in get_previous_quarters(period, year, n=args.n_history):
        label = f"{prev_period} {prev_year}"
        stats = _transcript_to_stats(ticker, prev_period, prev_year, label)
        if stats:
            stats["label"] = label
            historical_stats.append(stats)
            qoq_data.append(stats)

    # Compute
    print()
    nsi = compute_nsi(current_stats, historical_stats)
    print(f"NSI sigma:   {nsi['nsi_sigma']:+.2f}")
    print(f"Direction:   {nsi['direction']}")
    print(f"Δ sentiment: {nsi['delta_sentiment']:+.3f}  Δ hedge: {nsi['delta_hedge']:+.2f}")
    print(f"Based on {nsi.get('n_quarters', len(historical_stats))} historical quarters")

    # Write back
    with open(demo_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["nsi"]      = nsi
    data["qoq_data"] = qoq_data

    with open(demo_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nSaved → {demo_json}")


if __name__ == "__main__":
    main()
