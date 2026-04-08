"""
Speech processing module.
Handles converting audio into text and identifying speakers.
- Uses Whisper to transcribe earnings call audio
- Uses pyannote to separate different speakers
Output: clean transcript + speaker segments
"""

import whisper
import os
import torch
import spacy
import streamlit as st
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

def resolve_speaker_names(diarized_segments, av_turns, title_map=None):
    """
    Map speaker labels to real names from an Alpha Vantage transcript.
    Two modes:
    - SPEAKER_XX labels (pyannote ran): majority-vote per label across all segments.
    - UNKNOWN labels (pyannote unavailable/cached): direct per-segment word-overlap match.
    Appends title (CEO/CFO etc.) when found in title_map.
    """
    from collections import defaultdict, Counter

    if not av_turns:
        return diarized_segments

    import re as _re

    def _norm_words(text):
        """Lowercase and strip punctuation from words for robust matching."""
        return set(_re.sub(r"[^a-z0-9\s]", "", text.lower()).split())

    av_word_sets = [
        {"name": t["name"], "words": _norm_words(t["text"])}
        for t in av_turns
    ]

    def _best_match(seg_words):
        best_name, best_overlap = None, 0
        for av in av_word_sets:
            overlap = len(seg_words & av["words"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_name = av["name"]
        return best_name if best_overlap >= 2 else None

    def _with_title(name):
        if title_map and name in title_map:
            return f"{name} ({title_map[name]})"
        return name

    all_unknown = all(s.get("speaker", "UNKNOWN") == "UNKNOWN" for s in diarized_segments)

    if all_unknown:
        # No pyannote labels -- assign names directly per segment via word overlap
        result = []
        for seg in diarized_segments:
            seg_words = _norm_words(seg.get("text", ""))
            name = _best_match(seg_words) if len(seg_words) >= 2 else None
            result.append({**seg, "speaker": _with_title(name) if name else "UNKNOWN"})
        return result

    # SPEAKER_XX mode -- majority vote per label
    label_votes = defaultdict(Counter)
    for seg in diarized_segments:
        label = seg.get("speaker", "UNKNOWN")
        if label == "UNKNOWN":
            continue
        seg_words = _norm_words(seg.get("text", ""))
        if len(seg_words) < 2:
            continue
        name = _best_match(seg_words)
        if name:
            label_votes[label][name] += 1

    label_to_name = {}
    for label, votes in label_votes.items():
        name = votes.most_common(1)[0][0]
        label_to_name[label] = _with_title(name)

    return [
        {**seg, "speaker": label_to_name.get(seg.get("speaker", "UNKNOWN"), seg.get("speaker", "UNKNOWN"))}
        for seg in diarized_segments
    ]


_EXEC_TITLES = {
    "ceo", "cfo", "coo", "cso", "cto", "cio", "president", "chairman", "chair",
    "chief", "officer", "director", "founder", "vice president", "vp",
    "investor relations",
}
_ANALYST_MARKERS = {
    "analyst", "research",
    "goldman sachs", "morgan stanley", "j.p. morgan", "jpmorgan", "bank of america",
    "bofa", "barclays", "ubs", "citi", "citigroup", "wells fargo", "credit suisse",
    "deutsche bank", "raymond james", "oppenheimer", "cowen", "needham", "jefferies",
    "piper sandler", "baird", "rbc", "td securities", "bernstein", "truist", "mizuho",
    "bmo", "stifel", "cantor fitzgerald", "macquarie", "keybanc", "evercore",
    "guggenheim", "rosenblatt",
}
_EXCLUDE_LABELS = {"operator", "moderator", "conference", "coordinator"}


def is_management_speaker(name):
    """
    Classify a resolved speaker name as management, analyst/external, or unknown.

    Returns:
      True  — confirmed management/executive (name contains an exec title)
      False — confirmed non-management (operator, analyst, or known research firm)
      None  — ambiguous (plain name with no title, e.g. unresolved SPEAKER_XX)
    """
    if not name or name.upper() == "UNKNOWN":
        return False
    lower = name.lower()
    if lower.startswith("speaker_"):
        return None  # unresolved pyannote label
    for marker in _EXCLUDE_LABELS:
        if marker in lower:
            return False
    for marker in _ANALYST_MARKERS:
        if marker in lower:
            return False
    for title in _EXEC_TITLES:
        if title in lower:
            return True
    return None  # resolved name but no title info — ambiguous


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
