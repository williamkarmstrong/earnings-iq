"""
Data ingestion module.

This file is responsible for collecting earnings call data.
- Fetches transcripts from the Alpha Vantage API (fallback)
- Can be extended to load audio files if available

Think of this as the "data input" stage of the system.
"""

import streamlit as st
import requests
import yt_dlp
from dotenv import load_dotenv
import os

load_dotenv()

# Use .get() so a missing key doesn't crash the app at import time.
# Set to "" in secrets.toml to disable and preserve API credits during testing.
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")

@st.cache_data(show_spinner=False)
def fetch_transcript_cached(ticker, period, year):
    """
    Fetch an Alpha Vantage transcript with local file caching.
    Checks cache/{TICKER}_{YEAR}_{PERIOD}_transcript.txt first -- no API call if found.
    Saves to that path on first successful fetch so repeat runs are instant.
    Returns (transcript_text_string, error_message).
    """
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_{year}_{period}_transcript.txt")

    # Return from local cache if it contains speaker-formatted lines (Name: text).
    # Old cache files (content-only, no speaker names) are re-fetched once so
    # speaker attribution can work. Files without an API key are used as-is.
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = f.read()
        import re as _re
        has_speaker_format = bool(_re.search(r'^[A-Z][A-Za-z \.]{2,40}:\s', cached, _re.MULTILINE))
        if has_speaker_format or not ALPHA_VANTAGE_API_KEY:
            return cached, None
        # Old format detected and API key available -- fall through to re-fetch

    raw, err = fetch_backup_transcript(ticker, period, year)
    if raw is None:
        return None, err

    # AV returns a list of {speaker, content} dicts -- format as "Speaker: text"
    # so parse_av_speakers can extract names. Plain-string fallback kept for safety.
    if isinstance(raw, list):
        lines = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            speaker = (item.get("speaker") or item.get("name") or "").strip()
            content = (item.get("content") or item.get("text") or "").strip()
            if speaker and content:
                lines.append(f"{speaker}: {content}")
            elif content:
                lines.append(content)
        text = "\n".join(lines)
    else:
        text = str(raw)

    if text.strip():
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(text)
        return text, None

    return None, "Alpha Vantage returned an empty transcript."


def fetch_backup_transcript(ticker, period, year):
    """Fallback: fetch text transcript from Alpha Vantage API or return error message."""
    if not ALPHA_VANTAGE_API_KEY:
        return None, "No Alpha Vantage API key found."
        
    url = f"https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol={ticker}&quarter={year}{period}&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "transcript" in data:
                return data["transcript"], None
            else:
                return None, f"Alpha Vantage returned no transcript"
        else:
            return None, f"Alpha Vantage request failed with status code: {response.status_code}"
    except Exception as e:
        return None, f"Error fetching from Alpha Vantage: {e}"

def fetch_audio(ticker, period, year):
    """Attempt to fetch from local cache, otherwise search, score and download the best match."""
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    filename = f"{ticker}_{year}_{period}"
    expected_path_base = os.path.join(cache_dir, filename)
    
    # Check cache first
    for ext in ['.m4a', '.mp3', '.webm', '.wav', '.opus', '.ogg']:
        cached_file = expected_path_base + ext
        if os.path.exists(cached_file):
            return cached_file, f"cache/{filename}{ext}"

    def score_video(entry):
        # Ensure duration is over 25 mins to filter out short summaries/interviews
        duration = entry.get('duration')
        if duration and duration < 1500:
            return -1
            
        title = entry.get('title', '').lower()
        score = 0
        
        # Strongly weight the ticker
        t = str(ticker).lower()
        if t in title:
            score += 15
            
        # Weight year and period
        if str(year) in title:
            score += 15
        if str(period).lower() in title:
            score += 15
            
        # Slightly weight generic earnings keywords
        for k in ['earnings', 'call', 'conference']:
            if k in title:
                score += 5
                
        return score

    search_query = f"ytsearch5:${ticker} {period} {year} earnings conference call"
    
    try:
        # Step 1: Search and Score without downloading
        with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True}) as ydl_search:
            search_results = ydl_search.extract_info(search_query, download=False)
            
            best_match = None
            highest_score = -1
            
            if search_results and 'entries' in search_results:
                for entry in search_results['entries']:
                    if not entry:
                        continue
                    score = score_video(entry)
                    
                    # Require minimum score (45 ensures ticker, year, period are matched)
                    if score > highest_score and score > 45:
                        highest_score = score
                        best_match = entry
            
            if not best_match:
                return None, "No suitable audio match found."

        # Step 2: Download the winner
        video_url = best_match.get('original_url') or best_match.get('webpage_url') or best_match.get('url')
        if not video_url and best_match.get('id'):
            video_url = f"https://www.youtube.com/watch?v={best_match['id']}"
            
        if not video_url:
            return None, "Could not find a valid video URL."
            
        outtmpl = os.path.join(cache_dir, f"{filename}.%(ext)s")

        ydl_opts_dl = {
            'format': 'bestaudio/best',
            'outtmpl': outtmpl,
            'quiet': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts_dl) as ydl_dl:
            ydl_dl.download([video_url])
            
        # Find the path of the downloaded file
        for ext in ['.m4a', '.mp3', '.webm', '.wav', '.opus', '.ogg']:
            dl_file = expected_path_base + ext
            if os.path.exists(dl_file):
                return dl_file, f"{best_match['title']}"
                
    except Exception as e:
        return None, f"Error fetching audio: {e}"
    
    return None, "Unknown error during YouTube ingestion."
