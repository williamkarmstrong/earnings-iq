import numpy as np
import torchaudio
import sys
import re as _re
from collections import defaultdict, Counter
from app.event_study import get_earnings_date
import yfinance as yf

# --- REQUIRED MONKEY PATCHES ---
if not hasattr(np, "NaN"): np.NaN = np.nan
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda x: None

# --- THE TESTS ---

from app.speech import resolve_speaker_names, is_management_speaker

def test_analyst_mapping_logic():
    """
    Test 1: Verify that the resolve_speaker_names function 
    correctly appends (Analyst) based on the Alpha Vantage title.
    """
    print("\n--- TEST 1: ANALYST MAPPING ---")
    
    # Mock AV data where the title is specifically "Analyst"
    av_turns = [
        {"name": "Elon Musk", "text": "We are building the future of autonomy with FSD."},
        {"name": "Amit Daryanani", "text": "I guess my question is on the gross margin trajectory."}
    ]
    
    # This is what your parse_av_speakers function produces
    title_map = {
        "Elon Musk": "CEO",
        "Amit Daryanani": "Analyst"
    }

    # Mock Diarized Segments from Pyannote
    diarized_segments = [
        {"start": 10, "end": 40, "speaker": "SPEAKER_01", "text": "Building the future of autonomy and FSD."},
        {"start": 45, "end": 60, "speaker": "SPEAKER_02", "text": "I guess my question is on the gross margin trajectory."}
    ]

    resolved = resolve_speaker_names(diarized_segments, av_turns, title_map)
    
    success = True
    for seg in resolved:
        print(f"Resolved: {seg['speaker']}")
        if "Amit Daryanani" in seg['speaker'] and "(Analyst)" not in seg['speaker']:
            success = False
            
    if success:
        print("✅ SUCCESS: Analyst title correctly appended.")
    else:
        print("❌ FAIL: Analyst title missing or incorrect.")


def test_management_classification():
    """
    Test 2: Verify that is_management_speaker correctly 
    separates Management from Analysts/Operators.
    """
    print("\n--- TEST 2: MANAGEMENT CLASSIFICATION ---")
    
    test_cases = [
        ("Elon Musk (CEO)", True),
        ("Vaibhav Taneja (CFO)", True),
        ("Amit Daryanani (Analyst)", False),
        ("Dan Levy (Analyst)", False),
        ("Operator (Operator)", False),
        ("SPEAKER_01", False),
        ("Unknown", False)
    ]

    all_passed = True
    for name, expected in test_cases:
        result = is_management_speaker(name)
        status = "PASS" if result == expected else "FAIL"
        print(f"Name: {name:25} | Result: {str(result):5} | Expected: {str(expected):5} -> {status}")
        if result != expected:
            all_passed = False

    if all_passed:
        print("✅ SUCCESS: Management classification is deterministic.")
    else:
        print("❌ FAIL: Classification logic error.")

def test_earnings_call_date():
    test_cases = [
        ("TSLA",  "Q2", 2023),
        ("AAPL",  "Q1", 2024),
        ("MSFT",  "Q3", 2024),
        ("AMZN",  "Q4", 2023),
        ("GOOGL", "Q1", 2024),
        ("CRM",   "Q1", 2025),  # fiscal year starts Feb — Q1 reports in May
        ("CAT",   "Q1", 2024),  # non-standard fiscal calendar
        ("LIN",   "Q1", 2024),  # non-standard fiscal calendar
    ]

    print("\n--- TEST: EARNINGS DATE FETCH ---")
    for ticker, period, year in test_cases:
        print(f"\n{'='*60}")
        print(f"Ticker: {ticker}  |  Period: {period}  |  Year: {year}")

        ed = yf.Ticker(ticker).get_earnings_dates(limit=20)
        if ed is None or ed.empty:
            print("  [no data returned by yfinance]")
            continue

        # Strip tz so dates print cleanly
        idx = ed.index.tz_localize(None) if ed.index.tz is not None else ed.index
        ed.index = idx

        # Show all rows for target year ±1 so fiscal-year boundary cases are visible
        mask = ed.index.year.isin([year - 1, year, year + 1])
        subset = ed[mask].sort_index()
        print(f"  Raw yfinance rows (year ±1):\n{subset.to_string()}")

        result = get_earnings_date(ticker, period, year)
        print(f"  get_earnings_date() -> {result}")

if __name__ == "__main__":
    test_earnings_call_date()