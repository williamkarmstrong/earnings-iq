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
import re as _re
from dotenv import load_dotenv
import os

load_dotenv()

# Use .get() so a missing key doesn't crash the app at import time.
# Set to "" in secrets.toml to disable and preserve API credits during testing.
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")

import json
import os

def _parse_txt_transcript(txt_path):
    """
    Convert a legacy 'SpeakerName: text' transcript file into the av_turns format
    expected by parse_av_speakers(). Infers CEO/CFO/COO titles from introductory turns
    so is_management_speaker() can filter analysts out of the speaker panel.
    """
    _TITLE_RE = [
        _re.compile(r'our\s+(CEO|CFO|COO|President|Director|Chairman),?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', _re.I),
        _re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}),?\s+(?:our\s+)?(CEO|CFO|COO|President|Director|Chairman)\b', _re.I),
        _re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}),\s+Corporate Vice President[^,\n]*', _re.I),
    ]

    turns = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            m = _re.match(r'^([A-Za-z][^:]{2,60}):\s+(.+)', line.strip())
            if m:
                turns.append({"speaker": m.group(1).strip(), "title": "", "content": m.group(2).strip()})

    if not turns:
        return None

    # Infer exec titles from first 8 turns (operator intro + IR speaker typically name executives)
    name_title: dict[str, str] = {}
    for turn in turns[:8]:
        text = turn["content"]
        for pat in _TITLE_RE[:2]:
            for match in pat.finditer(text):
                if pat.pattern.startswith("our"):
                    name, title = match.group(2).strip(), match.group(1).strip().upper()
                else:
                    name, title = match.group(1).strip(), match.group(2).strip().upper()
                name_title[name] = title
        # Handle "Corporate Vice President" pattern
        for match in _TITLE_RE[2].finditer(text):
            name_title[match.group(1).strip()] = "Corporate Vice President, Investor Relations"

    for turn in turns:
        spk = turn["speaker"]
        if spk in name_title:
            turn["title"] = name_title[spk]
            continue
        # Partial last-name match (e.g. "Pat" → "Patrick Gelsinger")
        spk_last = spk.split()[-1]
        for full_name, title in name_title.items():
            if spk_last in full_name.split():
                turn["title"] = title
                break

    return turns


@st.cache_data(show_spinner=False)
def fetch_transcript(ticker, period, year):
    """
    Fetch an Alpha Vantage transcript and preserve JSON structure in cache.
    Falls back to a legacy .txt cache file before hitting the API.
    Returns: (list_of_dicts, error_message)
    """
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_{year}_{period}_transcript.json")
    txt_file   = os.path.join(cache_dir, f"{ticker}_{year}_{period}_transcript.txt")

    # 1. Check JSON cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as transcript:
                data = json.load(transcript)
            return data, None
        except Exception:
            os.remove(cache_file)

    # 2. Parse legacy .txt cache (avoids unnecessary API call)
    if os.path.exists(txt_file):
        try:
            turns = _parse_txt_transcript(txt_file)
            if turns:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(turns, f)
                return turns, None
        except Exception:
            pass

    # 3. Fetch from Alpha Vantage API
    raw, err = fetch_av_transcript(ticker, period, year)
    if raw is None:
        return None, err

    # 4. Save & Return
    try:
        with open(cache_file, "w", encoding="utf-8") as transcript:
            json.dump(raw, transcript)
            return raw, None
    except Exception as e:
        return None, f"Error saving transcript: {e}"

def fetch_av_transcript(ticker, period, year):
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
            # AV returns {"Note": "..."} or {"Information": "..."} on rate limit
            if "Note" in data or "Information" in data:
                msg = data.get("Note") or data.get("Information", "")
                return None, f"rate_limited:{msg}"
            return None, "no_transcript"
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
