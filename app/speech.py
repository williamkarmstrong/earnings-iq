"""
Speech processing module.

This file handles converting audio into text and identifying speakers.
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
from pyannote.audio import Model

# Use Streamlit caching so that if the user tweaks the UI, 
# the 50-minute file doesn't have to be transcribed again.
@st.cache_data(show_spinner=False)
def transcribe_audio(audio_path):
    """
    Transcribes audio using OpenAI's Whisper model.
    Returns the transcription result which includes text and segments.
    """
    # Hardware acceleration check for macOS (MPS)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    model = whisper.load_model("base", device=device)
    result = model.transcribe(audio_path)
    return result


@st.cache_data(show_spinner=False)
def map_speakers(audio_path, transcription_result):
    """
    Performs speaker diarization using pyannote.audio and maps it to Whisper segments.
    Ensure you have set the HUGGING_FACE_API_KEY environment variable. 
    """
    hf_token = st.secrets["HUGGING_FACE_API_KEY"]
    
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        
        # Send pyannote to MPS for faster processing on Mac
        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
            
    except Exception as e:
        print(f"Error loading pyannote: {e}")
        return transcription_result.get("segments", [])

    # Run diarization on the audio path
    diarization = pipeline(audio_path)
    
    segments = transcription_result.get("segments", [])
    mapped_segments = []
    
    for segment in segments:
        segment_start = segment["start"]
        segment_end = segment["end"]
        
        # Calculate overlaps between whisper segment and pyannote diarization tracks
        speaker_durations = {}
        for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
            overlap_start = max(segment_start, turn.start)
            overlap_end = min(segment_end, turn.end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > 0:
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + overlap
                
        # Assign the speaker with the maximum overlap for that segment
        if speaker_durations:
            best_speaker = max(speaker_durations, key=speaker_durations.get)
        else:
            best_speaker = "UNKNOWN"
            
        segment["speaker"] = best_speaker
        mapped_segments.append(segment)
        
    return mapped_segments


@st.cache_data(show_spinner=False)
def tokenize_audio_text(text):
    """
    Tokenizes the transcribed text using spaCy.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return [token.text for token in doc]
    except OSError:
        return text.split()
