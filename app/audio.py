"""
Audio analysis module.
Extracts two types of features from the earnings call audio:
  1. Librosa acoustic features -- pitch, energy, pauses.
  2. Wav2Vec2 embeddings -- deep speech representations.

All functions load audio in short windows by seeking directly in the file
rather than loading the full call into memory. A 90-minute WAV at 16kHz
would require ~1.7 GiB if loaded whole; window-based loading keeps peak
usage under 200 MB regardless of call length.
"""

import numpy as np
import librosa
import torch
import streamlit as st
from transformers import Wav2Vec2Processor, Wav2Vec2Model


@st.cache_resource(show_spinner="Loading Wav2Vec2 speech model…")
def _load_wav2vec2():
    """Load Wav2Vec2 once per server session via st.cache_resource."""
    try:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model     = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        model.eval()
        return processor, model, True
    except Exception as e:
        print(f"Wav2Vec2 unavailable: {e} -- confidence proxy will default to 0.5")
        return None, None, False


_processor, _wav2vec2, _wav2vec2_available = _load_wav2vec2()


def _file_duration(audio_path):
    """Return total duration in seconds without loading audio into memory."""
    try:
        return librosa.get_duration(path=audio_path)
    except TypeError:
        try:
            return librosa.get_duration(filename=audio_path)
        except Exception:
            return 3600.0  # safe fallback: assume 60 min
    except Exception:
        return 3600.0


def _load_window(audio_path, offset_s, duration_s, sr=16000):
    """Load a single time window from the file without reading the whole thing."""
    chunk, _ = librosa.load(
        audio_path,
        sr=sr,
        mono=True,
        offset=float(offset_s),
        duration=float(duration_s),
    )
    return chunk, sr


def _window_offsets(total_s, window_s, n=3):
    """Return n evenly-spread start offsets that don't exceed total_s."""
    offsets = [
        0.0,
        max(0.0, total_s / 2 - window_s / 2),
        max(0.0, total_s - window_s),
    ]
    return offsets[:n]


def extract_acoustic_features(audio_path):
    """
    Extract interpretable acoustic features using librosa.
    Loads three short windows (start, middle, end) rather than the full file.
    Returns a dict of scalar features.
    """
    sr        = 16000
    total_s   = _file_duration(audio_path)
    pitch_win = 20.0    # seconds per pitch window
    feat_win  = 120.0   # seconds per energy/pause window

    # ── Pitch (3 × 20s windows) ────────────────────────────────────────────
    valid_f0 = []
    for off in _window_offsets(total_s, pitch_win):
        try:
            chunk, _ = _load_window(audio_path, off, pitch_win, sr)
            if len(chunk) < sr:
                continue
            f0, _, _ = librosa.pyin(
                chunk,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
            )
            valid_f0.extend(f0[~np.isnan(f0)])
        except Exception:
            continue
    valid_f0   = np.array(valid_f0)
    pitch_mean = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0.0
    pitch_std  = float(np.std(valid_f0))  if len(valid_f0) > 0 else 0.0

    # ── Energy, pause ratio, tempo (3 × 120s windows) ──────────────────────
    energy_means, pause_ratios, tempos = [], [], []
    for off in _window_offsets(total_s, feat_win):
        try:
            chunk, _ = _load_window(audio_path, off, feat_win, sr)
            if len(chunk) < sr:
                continue
            rms = librosa.feature.rms(y=chunk)[0]
            e_mean = float(np.mean(rms))
            energy_means.append(e_mean)
            threshold = e_mean * 0.20
            pause_ratios.append(float(np.sum(rms < threshold) / len(rms)))
            t, _ = librosa.beat.beat_track(y=chunk, sr=sr)
            tempos.append(float(t) if np.isscalar(t) else float(t[0]))
        except Exception:
            continue

    energy_mean = float(np.mean(energy_means)) if energy_means else 0.0
    pause_ratio = float(np.mean(pause_ratios)) if pause_ratios else 0.0
    tempo       = float(np.mean(tempos))        if tempos       else 0.0

    return {
        "pitch_mean":  round(pitch_mean, 2),
        "pitch_std":   round(pitch_std, 2),
        "tempo":       round(tempo, 2),
        "energy_mean": round(energy_mean, 4),
        "pause_ratio": round(pause_ratio, 4),
    }


def extract_wav2vec2_features(audio_path, window_s=200.0):
    """
    Extract Wav2Vec2 deep speech embeddings from three windows.
    Each window is loaded independently from disk — no full-file load.
    Returns a dict with confidence proxy and raw norm.
    """
    if not _wav2vec2_available:
        return {"embedding_mean": [], "confidence_proxy": 0.5}

    sr      = 16000
    total_s = _file_duration(audio_path)

    embeddings     = []
    all_step_norms = []

    with torch.no_grad():
        for off in _window_offsets(total_s, window_s):
            try:
                chunk, _ = _load_window(audio_path, off, window_s, sr)
                if len(chunk) < sr:
                    continue
                inputs  = _processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
                outputs = _wav2vec2(**inputs)
                hidden  = outputs.last_hidden_state.squeeze(0)       # (seq_len, 768)
                all_step_norms.extend(torch.norm(hidden, dim=-1).numpy().tolist())
                embeddings.append(hidden.mean(dim=0).numpy())
            except Exception:
                continue

    if not embeddings:
        return {"embedding_mean": [], "confidence_proxy": 0.5}

    mean_embedding = np.mean(embeddings, axis=0)
    raw_norm       = float(np.linalg.norm(mean_embedding))

    step_arr = np.array(all_step_norms)
    cv = float(np.std(step_arr)) / (float(np.mean(step_arr)) + 1e-6)
    confidence_proxy = float(np.tanh(cv / 0.6))

    return {
        "embedding_mean":    mean_embedding.tolist(),
        "confidence_proxy":  round(confidence_proxy, 4),
        "wav2vec2_raw_norm": round(raw_norm, 3),
    }


@st.cache_data(show_spinner=False)
def extract_audio_features(audio_path):
    """
    Main entry point — runs acoustic and Wav2Vec2 extraction.
    Cached so re-runs on the same file are instant.
    """
    acoustic = extract_acoustic_features(audio_path)
    deep     = extract_wav2vec2_features(audio_path)
    return {**acoustic, **deep}
