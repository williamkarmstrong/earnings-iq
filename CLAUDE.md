# CLAUDE.md — CFA AI Investment Challenge

## Project Overview

This project is a multimodal earnings call analysis platform built for the CFA AI Investment Challenge. It analyses earnings conference call **audio and transcripts** to extract actionable investment signals — combining speech emotion recognition with NLP-based sentiment analysis to identify management confidence shifts, narrative changes, and emerging risk signals that text-only approaches miss.

The platform targets investment professionals who need to scale qualitative analysis across broad coverage universes without sacrificing signal quality.

---

## Architecture

The system is composed of four layers:

```
[Audio / Transcript Sources]
    Alpha Vantage API, yt-dlp, local audio files
          │
          ▼
[Speech AI Layer]
    Whisper (ASR + timestamp alignment)
    pyannote.audio (speaker diarization: CEO, CFO, analysts)
          │
          ▼
[Tone / Sentiment Analysis — runs in parallel]
    Tone:      Librosa (pitch variance, spectral centroid, tempo)
               Wav2Vec2-BERT (speech emotion recognition)
    Sentiment: spaCy + FinBERT (financial sentiment classification)
          │
          ▼
[Multimodal AI Fusion Engine]
    scikit-learn, Pandas, NumPy
    Composite Confidence Score = f(FinBERT sentiment, acoustic pitch variance)
    Narrative Shift Index (cross-quarter comparison)
    Risk Signal Detection
          │
          ▼
[Frontend Dashboard — Streamlit + Plotly]
    Sentiment-over-time trajectory
    Management Confidence Score
    Narrative Change Index
    Risk Signal Level
    Insight Generation Agent (summary output)
```

### Graceful Degradation

If audio cannot be retrieved, the system falls back to **transcript-only analysis** via the Alpha Vantage API. All acoustic features are skipped; FinBERT sentiment analysis proceeds as normal. The UI should clearly indicate which mode is active.

---

## Key Modules

| Module | Purpose | Libraries |
|---|---|---|
| `audio_retrieval` | Fetch earnings call audio | `yt-dlp`, Alpha Vantage API |
| `transcription` | ASR with timestamp alignment | `openai-whisper` |
| `diarization` | Speaker separation (CEO/CFO/analyst) | `pyannote.audio` |
| `acoustic_features` | Low-level audio feature extraction | `librosa` |
| `speech_emotion` | Speech emotion recognition | `Wav2Vec2-BERT` (Hugging Face) |
| `text_sentiment` | Financial sentiment classification | `spaCy`, `FinBERT` (Hugging Face) |
| `fusion_engine` | Combine acoustic + textual features | `scikit-learn`, `pandas`, `numpy` |
| `insight_agent` | Generate natural language summaries | LLM-based agent |
| `dashboard` | Visualisation and user interface | `streamlit`, `plotly` |

---

## Signals & Metrics

The platform computes the following output signals:

- **Composite Confidence Score** — weighted combination of FinBERT sentiment polarity and acoustic pitch variance, computed per speaker and per call section (prepared remarks vs. Q&A).
- **Narrative Shift Index** — cross-quarter delta in language patterns; flags calls where language deviates significantly from the prior period. Values >2 standard deviations above sector peers warrant investigation.
- **Risk Signal Level** — rule-based detection of hedging language, forward-looking statement changes, and acoustic stress markers.
- **Sentiment Trajectory** — time-series of sentiment across the call, enabling identification of sections where tone deteriorates (e.g., gross margin discussion vs. revenue discussion).

---

## Data Sources

| Source | Type | Usage |
|---|---|---|
| Alpha Vantage API | Transcript (text) | Primary fallback when audio unavailable |
| `yt-dlp` | Audio | Automated retrieval from public sources |
| Local audio files | Audio | Sample calls for demonstration |

**Important:** The platform operates exclusively on **publicly available information** (public earnings disclosures). No proprietary or non-public data sources should be integrated.

---

## Tech Stack

- **Frontend:** Streamlit, Plotly
- **Speech Processing:** OpenAI Whisper, pyannote.audio, Librosa
- **ML / NLP:** Hugging Face Transformers (Wav2Vec2-BERT, FinBERT), spaCy, scikit-learn
- **Data:** Pandas, NumPy
- **Audio Retrieval:** yt-dlp, Alpha Vantage API (via `requests`)
- **Language:** Python 3.10+

---

## Development Guidelines

### Code Style
- Follow PEP 8. Use type hints throughout.
- Modules should be loosely coupled — the fusion engine should accept either `(acoustic_features, text_features)` or `(text_features_only,)` to support graceful degradation cleanly.
- All signal scores should be in a normalised range (e.g. 0–100 or -1 to +1) with clear documentation of the scale.

### Explainability Requirements
Every signal or score surfaced to the user **must** be traceable to a source:
- Sentiment flags → reference the specific transcript segment
- Acoustic signals → reference the timestamp and speaker
- Do not surface opaque composite scores without showing the underlying contributors

This is both an ethical requirement and a usability one — analysts need to validate outputs before acting on them.

### Error Handling
- Audio retrieval failures should trigger graceful degradation automatically, logging the reason.
- Diarization failures (e.g. overlapping speakers, poor audio quality) should fall back to speaker-agnostic analysis rather than crashing.
- API rate limits from Alpha Vantage should be handled with exponential backoff.

### Testing
- Unit test the fusion engine with synthetic `(sentiment_score, acoustic_score)` pairs to verify score normalisation.
- Integration test the full pipeline on at least one locally stored sample call.
- Regression test narrative shift calculations across at least two consecutive quarters for a sample company.

---

## Ethical Constraints

These are hard constraints, not suggestions:

1. **Decision-support only.** The platform must never be framed as or used as an automated trading system. Outputs are inputs to human judgment, not replacements for it.
2. **Public data only.** No non-public or proprietary data sources. All earnings calls ingested must be public disclosures.
3. **Bias awareness.** Speech emotion recognition may misinterpret accents, speaking styles, or cultural cadence as hesitation or stress. Flag this limitation in the UI wherever acoustic signals are displayed.
4. **No overreliance.** The UI should encourage analysts to review the original transcript/audio where signals are flagged, not just accept the model's output.

---

## Example Use Case

> Nvidia Q4 2025 earnings call (February 2026):
> - Wav2Vec2-BERT detected a 23% drop in CFO vocal confidence during gross margin discussion
> - FinBERT flagged a shift from confident to hedged language in margin-related commentary vs. Q3
> - Narrative Shift Index: 2.1 standard deviations above semiconductor sector peers
> - Output: Risk signal raised, flagged for analyst review before market fully priced guidance revision

---

## References

- Price et al. (2012) — Textual tone in earnings calls predicts abnormal returns
- Chen et al. (2023) — Audio contains richer sentiment signals than transcripts alone, especially in Q&A
- Hajek & Munk (2023) — Combining speech emotion recognition + text sentiment improves financial distress prediction
- Huang et al. (2025) — Earnings call transcripts require significant manual effort; key insights benefit from automated extraction
- Cao et al. (2024) — LLMs extract fine-grained, predictive signals from earnings calls for volatility forecasting
- CFA Institute (2023) — *Handbook of AI and Big Data Applications in Investments*, Part II
