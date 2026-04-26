import numpy as np
import torchaudio
import sys
import re as _re
from collections import defaultdict, Counter

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

if __name__ == "__main__":
    test_analyst_mapping_logic()
    test_management_classification()