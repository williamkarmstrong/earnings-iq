# Earnings-IQ
## Overview
The proposed solution allows users to interact with an AI-powered analysis dashboard that automatically collects, processes, and analyses company earnings calls. The platform will ingest both earnings call audio and transcripts, enabling deeper analysis of not only what executives say but how they say it. By combining speech processing with language analysis, the tool reduces manual analysis time while providing unique insights such as cross-company comparisons and highlighting significant changes in tone or sentiment across reporting periods.

## Key Features
- Audio Transcription (using Whisper model)
- Alpha Vantage API for Earnings Call Transcripts (graceful degradation)
- Speaker Diarization (who said what)
- Sentiment Analysis (positive / negative / neutral using FinBERT)
- Contextual Insights (management tone, confidence shifts)
- Data Output (JSON / CSV for dashboards)
- Dashboard (powered by Streamlit)

## Tech Stack
- Python
- OpenAI Whisper
- PyTorch
- Speaker diarization (e.g. pyannote)
- NLP libraries (NLTK / transformers)
- Pandas for data processing
  
## Installation
1. git clone https://github.com/williamkarmstrong/earnings-iq
2. cd earnings-iq
3. pip install -r requirements.txt

## Usage
1. streamlit run app.py

## Live Demo
https://earningsiq.streamlit.app
