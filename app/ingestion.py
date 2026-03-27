"""
Data ingestion module.

This file is responsible for collecting earnings call data.
- Fetches transcripts from the Alpha Vantage API (fallback)
- Can be extended to load audio files if available

Think of this as the "data input" stage of the system.
"""

import requests

def fetch_earnings_data(ticker, period, year):
    pass

def fetch_audio_url(ticker, period, year):
    pass
