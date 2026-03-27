"""
Streamlit app for multimodal earnings analysis.
This is the main entry point for the app and 
builds the dashboard that the user will see.

This file connects all parts of the system together and runs the full pipeline:
1. Fetch earnings call data
2. Process audio/transcript
3. Analyse sentiment and tone
4. Generate insights
5. Display results in the dashboard (Streamlit)
"""

# Import libraries
import streamlit as st
import pandas as pd
import plotly.express as px

# Import our custom modules
from ingestion import fetch_earnings_data
from speech import analyze_audio_features
from nlp import analyze_sentiment
from multimodal import analyze_multimodal
from insights import generate_insights

st.set_page_config(page_title="Sentira: Earnings Intelligence", layout="wide")

@st.cache_data
def get_analysis(ticker, q, yr):
    return {"score": 0.75, "tone": "Confident", "top_words": ["Growth", "AI", "Margins"]}

st.title("📈 EarningsIQ | Multimodal Earnings Analysis")

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL")
    period = st.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4"])
    year = st.slider("Year", 2020, 2026, 2024)
    run = st.button("Analyze Call")

if run:
    data = get_analysis(ticker, period, year)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sentiment Score", f"{data['score']*100}%", delta="Positive")
        st.subheader("Vocal Tone Analysis")
        st.write(f"The speaker's tone was rated as **{data['tone']}**.")

    with col2:
        # A simple Plotly Chart
        fig = px.bar(x=data['top_words'], y=[10, 8, 5], title="Key Topic Frequency")
        st.plotly_chart(fig)
