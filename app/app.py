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
import time

# Import our custom modules (Commented out for now to avoid dependency issues during UI verification)
from ingestion import fetch_earnings_data, fetch_audio_url
from speech import analyse_audio_features
from nlp import analyse_sentiment
from multimodal import analyse_multimodal
from insights import generate_insights

st.set_page_config(page_title="EarningsIQ", layout="wide")

@st.cache_data
def get_analysis(ticker, q, yr):
    # Mock data for demonstration
    return {
        "score": 0.75, 
        "tone": "Confident", 
        "top_words": [
            {"Word": "Growth", "Count": 10}, 
            {"Word": "AI", "Count": 8}, 
            {"Word": "Margins", "Count": 5}
        ],
        "sentiment_history": [
            {"Date": "2023-Q1", "Score": 0.65},
            {"Date": "2023-Q2", "Score": 0.70},
            {"Date": "2023-Q3", "Score": 0.68},
            {"Date": "2023-Q4", "Score": 0.75},
            {"Date": "2024-Q1", "Score": 0.72},
        ],
        "signals": {
            "management_confidence": {"score": 85, "delta": "High", "delta_arrow": "up"},
            "narrative_change": {"narrative": "Moderate", "delta_arrow": "up"},
            "risk_level": {"risk": "Low", "delta_arrow": "up"}
        },
        "insight_agent_response": (
            "The management team maintains a highly confident outlook, particularly regarding the scaling of AI infrastructure. "
            "While there is a moderate shift in narrative towards margin optimization compared to last year's focus on pure growth, "
            "the overall risk profile remains low due to strong cash flows and a robust backlog. "
            "Key areas to monitor include the timing of Capex returns and competitive pricing in the cloud segment."
        )
    }

st.title("📈 EarningsIQ | Multimodal Earnings Analysis")

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL")
    period = st.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4"])
    year = st.slider("Year", 2020, 2026, 2024)
    run = st.button("Analyze Call")

if run:
    # Simulated processing with a loading bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    stages = [
        ("Fetching earnings data...", 25),
        ("Processing audio and transcript...", 50),
        ("Analysing sentiment and vocal tone...", 75),
        ("Generating multimodal insights...", 100)
    ]
    
    for status, percentage in stages:
        status_text.text(status)
        time.sleep(0.5)  # Simulate work
        progress_bar.progress(percentage)
    
    # Clear processing UI
    status_text.empty()
    progress_bar.empty()

    data = get_analysis(ticker, period, year)

    # Top Row: Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sentiment Score", f"{data['score']*100}%", delta="7%")
    with col2:
        st.metric("Management Confidence", f"{data['signals']['management_confidence']['score']}%", delta=data['signals']['management_confidence']['delta'], delta_arrow=data['signals']['management_confidence']['delta_arrow'])
    with col3:
        st.metric("Narrative Change", data['signals']['narrative_change']['narrative'], delta="Neutral", delta_arrow=data['signals']['narrative_change']['delta_arrow'])
    with col4:
        st.metric("Risk Level", data['signals']['risk_level']['risk'], delta="Stable", delta_arrow=data['signals']['risk_level']['delta_arrow'])

    st.divider()

    # Middle Row: Sentiment Over Time & Top Topics
    col_mid1, col_mid2 = st.columns(2)
    with col_mid1:
        st.subheader("Sentiment Over Time")
        history_df = pd.DataFrame(data['sentiment_history'])
        fig_line = px.line(history_df, x="Date", y="Score", title="Trend Analysis", markers=True)
        fig_line.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_line, use_container_width=True)

    with col_mid2:
        st.subheader("Key Topic Frequency")
        top_words_df = pd.DataFrame(data['top_words'])
        fig_bar = px.bar(top_words_df, x="Word", y="Count", title="Mention Counts")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # Bottom Row: Vocal Tone & Agent Insights
    col_bot1, col_bot2 = st.columns([1, 2])
    with col_bot1:
        st.subheader("Vocal Tone Analysis")
        st.info(f"The speaker's tone was rated as **{data['tone']}**.")
        st.write("Confidence levels were sustained throughout the Q&A session, with notable stability in the CEO's responses.")

    with col_bot2:
        st.subheader("🤖 AI Insight Agent")
        st.markdown(f"> {data['insight_agent_response']}")
