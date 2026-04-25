"""
Speech processing module.
Handles converting audio into text and identifying speakers.
- Uses Whisper to transcribe earnings call audio
- Uses pyannote to separate different speakers
Output: clean transcript + speaker segments
"""

import sys
import dataclasses
import numpy as np
import torchaudio

# Compatibility patches must run before pyannote is imported
if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend: None
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
if not hasattr(torchaudio, "AudioMetaData"):
    @dataclasses.dataclass
    class _AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str
    torchaudio.AudioMetaData = _AudioMetaData

sys.modules['numpy'] = np
sys.modules['torchaudio'] = torchaudio

import whisper
import os
import torch
import spacy
import streamlit as st
import re as _re
from collections import defaultdict, Counter
from pyannote.audio import Pipeline

@st.cache_data(show_spinner=False)
def transcribe_audio(audio_path):
    """
    Transcribes audio using OpenAI's Whisper model.
    Returns the transcription result including text and segments.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = whisper.load_model("base", device=device)
    result = model.transcribe(audio_path)
    return result

@st.cache_data(show_spinner=False)
def map_speakers(audio_path, transcription_result):
    """
    Performs speaker diarization using pyannote and maps it to Whisper segments.
    Requires HUGGING_FACE_API_KEY in streamlit secrets.
    Falls back to raw Whisper segments (no speaker labels) if diarization fails.
    """
    # Graceful degradation: missing token or network failure -> return unlabelled segments
    hf_token = st.secrets.get("HUGGING_FACE_API_KEY", "")
    if not hf_token:
        print("No Hugging Face token -- skipping speaker diarization.")
        return transcription_result.get("segments", [])

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
    except Exception as e:
        print(f"Error loading pyannote pipeline: {e}")
        return transcription_result.get("segments", [])

    try:
        diarization = pipeline(audio_path)
    except Exception as e:
        print(f"Diarization inference failed: {e}")
        return transcription_result.get("segments", [])

    segments = transcription_result.get("segments", [])
    mapped_segments = []

    for segment in segments:
        segment_start = segment["start"]
        segment_end = segment["end"]
        speaker_durations = {}

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            overlap_start = max(segment_start, turn.start)
            overlap_end = min(segment_end, turn.end)
            overlap = max(0, overlap_end - overlap_start)
            if overlap > 0:
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + overlap

        best_speaker = max(speaker_durations, key=speaker_durations.get) if speaker_durations else "UNKNOWN"
        segment["speaker"] = best_speaker
        mapped_segments.append(segment)

    return mapped_segments

@st.cache_data(show_spinner=False)
def resolve_speaker_names(diarized_segments, av_turns, title_map=None):
    """
    Map speaker labels to names. If the speaker is an analyst, 
    the title will be simplified to 'Analyst' for easier filtering.
    """
    if not av_turns:
        return diarized_segments

    def _norm_words(text):
        stops = {'the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'for', 'on', 'with', 'welcome', 'thank'}
        words = _re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
        return set(w for w in words if w not in stops)

    # 1. Search Index
    av_word_sets = [
        {"name": t["name"], "words": _norm_words(t["text"]), "index": i}
        for i, t in enumerate(av_turns)
    ]

    def _best_match(seg_words, is_early_segment=False):
        best_name, best_overlap = None, 0
        for av in av_word_sets:
            overlap = len(seg_words & av["words"])
            if is_early_segment and av["index"] < 3:
                overlap *= 2.5 
            if overlap > best_overlap:
                best_overlap = overlap
                best_name = av["name"]
        return best_name if best_overlap >= 2 else None

    # 2. Simplified Title Helper
    def _with_title(name):
        if title_map and name in title_map:
            title = title_map[name]
            # If the title contains 'Analyst', force it to 'Analyst'
            if "analyst" in title.lower():
                return f"{name} (Analyst)"
            # Otherwise, use the executive title (CEO, CFO, etc.)
            return f"{name} ({title})"
        return name

    # 3. Weighted Voting
    label_votes = defaultdict(Counter)
    sorted_segs = sorted(diarized_segments, key=lambda x: x.get("start", 0))
    
    for seg in sorted_segs:
        label = seg.get("speaker", "UNKNOWN")
        if label == "UNKNOWN": continue
        
        seg_words = _norm_words(seg.get("text", ""))
        if len(seg_words) < 2: continue
        
        is_early = seg.get("start", 0) < 180
        name = _best_match(seg_words, is_early_segment=is_early)
        
        if name:
            weight = max(1, len(seg_words) // 5)
            label_votes[label][name] += weight

    # 4. Final Mapping
    label_to_name = {}
    for label, votes in label_votes.items():
        if not votes:
            label_to_name[label] = label
            continue

        winner_name = votes.most_common(1)[0][0]
        label_to_name[label] = _with_title(winner_name)

    # 5. Fallback: direct per-segment name matching when pyannote was unavailable.
    # label_votes is empty when all segments had no pyannote speaker label (UNKNOWN).
    # Try to match each segment's text directly against AV turns by word-overlap so
    # the speaker attribution panel is not silently empty when diarization is skipped.
    if not label_votes:
        result = []
        for seg in diarized_segments:
            seg_words = _norm_words(seg.get("text", ""))
            if len(seg_words) >= 2:
                is_early = seg.get("start", 0) < 180
                name = _best_match(seg_words, is_early_segment=is_early)
                if name:
                    result.append({**seg, "speaker": _with_title(name)})
                    continue
            result.append({**seg, "speaker": seg.get("speaker", "UNKNOWN")})
        return result

    return [
        {**seg, "speaker": label_to_name.get(seg.get("speaker", "UNKNOWN"), seg.get("speaker", "UNKNOWN"))}
        for seg in diarized_segments
    ]

def is_management_speaker(name):
    """
    Returns True for confirmed management, False for confirmed analysts/operators,
    None for unresolved or ambiguous speakers (no title information).
    """
    if not name or name in ("UNKNOWN", "") or name.upper().startswith("SPEAKER_"):
        return None  # unresolved pyannote label — ambiguous
    if "(" not in name:
        return None  # resolved name but no title — ambiguous, don't exclude

    title = name.split("(")[-1].split(")")[0].lower()

    if "analyst" in title or "operator" in title:
        return False

    exec_markers = ["ceo", "cfo", "coo", "vp", "president", "director", "chairman", "officer", "management"]
    if any(m in title for m in exec_markers):
        return True

    return None  # title present but unrecognised — ambiguous

@st.cache_data(show_spinner=False)
def tokenize_audio_text(text):
    """
    Tokenizes transcribed text using spaCy.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return [token.text for token in doc]
    except OSError:
        return text.split()
