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
    
    in_qa = False
    speaker_roles = {}
    
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
            
        text_lower = segment["text"].lower()
        
        # Roughly determine the type of the segment
        qa_triggers = ["open the line for questions", "open the call for questions", 
                       "first question", "question-and-answer", "q&a", "take questions"]
        if not in_qa and any(phrase in text_lower for phrase in qa_triggers):
            in_qa = True
            
        segment_type = "qa" if in_qa else "prepared"

        # Roughly determine the role of the speaker
        if best_speaker not in speaker_roles:
            # Assume SPEAKER_01 is usually the operator in pyannote diarization
            if best_speaker == "SPEAKER_01":
                speaker_roles[best_speaker] = "Operator"
            elif best_speaker == "SPEAKER_00":
                speaker_roles[best_speaker] = "CEO"
            elif not in_qa:
                # Speakers in the prepared remarks are usually executives
                speaker_roles[best_speaker] = "CFO"
            else:
                # New speaker in Q&A: if they ask a question or use introductory pleasantries, likely an analyst
                analyst_cues = ["?", "question", "thanks", "hi ", "hello", "morning", "afternoon"]
                if any(cue in text_lower for cue in analyst_cues):
                    speaker_roles[best_speaker] = "Analyst"
                else:
                    # Otherwise, it might be an executive answering a question for the first time
                    speaker_roles[best_speaker] = "Executive"
                
        # Update analyst role dynamically if they ask explicit questions later
        if in_qa and speaker_roles.get(best_speaker) == "Executive":
            if "my question" in text_lower or "thanks for taking" in text_lower:
                speaker_roles[best_speaker] = "Analyst"

        role = speaker_roles.get(best_speaker, "Unknown")

        processed_segment = {
            "id": segment["id"],
            "speaker": best_speaker,
            "role": role,
            "content": segment["text"],
            "start": segment["start"],
            "end": segment["end"],
            "source": "audio",
            "type": segment_type
        }

        mapped_segments.append(processed_segment)
        
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
