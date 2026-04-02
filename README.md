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
In your terminal run the following commands
1. git clone https://github.com/williamkarmstrong/earnings-iq
2. cd earnings-iq
3. pip install -r requirements.txt
4. create a .streamlit/secrets.toml file and using .streamlit/secrets.toml.example as a template, add your own API keys

## Usage
In your terminal run the command `streamlit run app/app.py` which will automatically open the the app in a browser at http://localhost:8501

## Live Demo
https://earningsiq.streamlit.app
(Note for team: this will update automatically when you push changes)

## Notes
`yt dlp` is used to download audio from YouTube, however it will not work
on the cloud hosted domain, due to YouTube only allowing downloads locally.
