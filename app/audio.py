"""
Audio analysis module.
Extracts two types of features from the earnings call audio:
  1. Librosa acoustic features -- pitch, tempo, energy, pauses.
     These are interpretable proxies for vocal confidence.
  2. Wav2Vec2 embeddings -- deep speech representations that capture
     hesitation and delivery patterns beyond what acoustics alone reveal.
     This is the differentiating layer described in the concept paper.
"""

import numpy as np
import librosa
import torch
import streamlit as st
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Load Wav2Vec2 once at module level to avoid reloading on every call.
# Wrapped in try/except so a missing or partially-downloaded model doesn't
# crash the app on import -- extract_wav2vec2_features degrades gracefully.
try:
    _processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    _wav2vec2  = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    _wav2vec2.eval()  # inference mode -- disables dropout
    _wav2vec2_available = True
except Exception as _e:
    print(f"Wav2Vec2 unavailable: {_e} -- confidence proxy will default to 0.5")
    _processor = None
    _wav2vec2  = None
    _wav2vec2_available = False


def _load_audio(audio_path):
    """
    Load audio and resample to 16kHz mono.
    Wav2Vec2 and librosa both expect 16kHz -- yt-dlp already outputs
    this format but we resample defensively in case of cached files.
    """
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    return y, sr


def extract_acoustic_features(audio_path):
    """
    Extract interpretable acoustic features using librosa.
    These are the surface-level vocal indicators -- useful for displaying
    to analysts as they map to intuitive concepts like speaking rate,
    pitch stability, and pause frequency.
    Returns a dict of scalar features.
    """
    y, sr = _load_audio(audio_path)

    # Pitch -- sample 3 × 20s windows (start, middle, end) for call-wide representation.
    # pyin is slow; three 20s windows totals 60s of analysis but covers more of the call
    # than the original first-60s-only approach.
    window_s = sr * 20
    total_len = len(y)
    pitch_chunks = [
        y[:window_s],
        y[max(0, total_len // 2 - window_s // 2): max(0, total_len // 2 - window_s // 2) + window_s],
        y[max(0, total_len - window_s):],
    ]
    valid_f0_all = []
    for _chunk in pitch_chunks:
        if len(_chunk) < sr:
            continue
        _f0, _, _ = librosa.pyin(
            _chunk,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7")
        )
        valid_f0_all.extend(_f0[~np.isnan(_f0)])
    valid_f0 = np.array(valid_f0_all)
    pitch_mean = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0.0
    pitch_std  = float(np.std(valid_f0))  if len(valid_f0) > 0 else 0.0

    # Tempo -- speaking rate in approximate BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    # RMS energy -- overall vocal intensity
    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_std  = float(np.std(rms))

    # Pause detection -- frames where RMS drops below 20% of mean energy.
    # 20% is more robust than 10% for compressed audio (yt-dlp m4a/webm) where codec
    # noise raises the apparent floor and causes over-classification of speech as silence.
    silence_threshold = energy_mean * 0.20
    silent_frames = np.sum(rms < silence_threshold)
    pause_ratio = float(silent_frames / len(rms)) if len(rms) > 0 else 0.0

    # MFCCs -- compact descriptor of vocal timbre
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfcc.mean(axis=1).tolist()

    return {
        "pitch_mean":  round(pitch_mean, 2),
        "pitch_std":   round(pitch_std, 2),
        "tempo":       round(tempo, 2),
        "energy_mean": round(energy_mean, 4),
        "energy_std":  round(energy_std, 4),
        "pause_ratio": round(pause_ratio, 4),
        "mfcc_means":  mfcc_means,
    }


def extract_wav2vec2_features(audio_path, max_duration_seconds=600):
    """
    Extract Wav2Vec2 deep speech embeddings.
    We cap at max_duration_seconds and sample from the beginning, middle,
    and end of the call -- full earnings calls are 45-90 mins which would
    exceed memory if processed whole.
    The three-section sampling captures prepared remarks, mid-call, and Q&A.
    Returns a dict with mean embedding and a scalar confidence proxy.
    """
    y, sr = _load_audio(audio_path)

    # Sample three windows across the call
    window = sr * max_duration_seconds // 3
    total  = len(y)

    segments = []
    for offset in [0, max(0, total // 2 - window // 2), max(0, total - window)]:
        chunk = y[offset: offset + window]
        if len(chunk) > sr:  # only include chunks longer than 1 second
            segments.append(chunk)

    # If model didn't load, return a neutral default
    if not _wav2vec2_available:
        return {"embedding_mean": [], "confidence_proxy": 0.5}

    embeddings = []
    with torch.no_grad():  # no gradients needed for inference
        for chunk in segments:
            inputs = _processor(
                chunk,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )
            outputs = _wav2vec2(**inputs)
            # Mean pool across the time dimension to get a fixed-size vector
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)

    if not embeddings:
        return {"embedding_mean": [], "confidence_proxy": 0.5}

    mean_embedding = np.mean(embeddings, axis=0)

    # Confidence proxy: embedding norm correlates with speech clarity.
    # Normalised to [0,1] using range 5–25 which reflects empirical Wav2Vec2-base
    # mean-pooled norms for speech (typically 8–20). The original 5–50 range produced
    # systematically low proxies (~0.1–0.3) for normal speech, dragging MCI down.
    raw_norm = float(np.linalg.norm(mean_embedding))
    confidence_proxy = float(np.clip((raw_norm - 5) / 20, 0.0, 1.0))

    return {
        "embedding_mean":   mean_embedding.tolist(),
        "confidence_proxy": round(confidence_proxy, 4),
        "wav2vec2_raw_norm": round(raw_norm, 3),
    }


@st.cache_data(show_spinner=False)
def extract_audio_features(audio_path):
    """
    Main entry point -- runs both acoustic and Wav2Vec2 extraction
    and returns a combined feature dict for downstream multimodal fusion.
    Cached so re-runs on the same file are instant.
    """
    acoustic = extract_acoustic_features(audio_path)
    deep     = extract_wav2vec2_features(audio_path)
    return {**acoustic, **deep}
