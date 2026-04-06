"""
Natural Language Processing (NLP) module.

This file analyses the transcript text.
- Cleans and processes text using spaCy
- Uses FinBERT to detect financial sentiment (positive, neutral, negative)

This helps understand what is being said in the earnings call.
"""

from transformers import pipeline
from bertopic import BERTopic
import spacy
import streamlit as st
import warnings

@st.cache_resource(show_spinner=False)
def load_finbert():
    # FinBERT is specifically engineered for financial text 
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

@st.cache_data(show_spinner=False)
def analyse_sentiment(mapped_segments):
    if not mapped_segments:
        return {
            "positive": 0, "negative": 0, "neutral": 0, "total": 0, "net_score": 0.0,
            "management_confidence": 0.0, "narrative_shift_index": 0.0, "qa_stress_indicator": 0.0,
            "individual_management_confidence": {"CEO": 50.0, "CFO": 50.0, "Executive": 50.0},
            "topic_decomposition": {}
        }, []
        
    finbert = load_finbert()
    
    # 1. Prepare Data
    docs = [s.get("content", "").strip() or "EMPTY SEGMENT" for s in mapped_segments]
    word_counts = [len(text.split()) for text in docs]
    
    # 2. Topic Modeling (Isolate the fragility)
    topics = [-1] * len(docs)
    topic_names = {-1: "General/Contextual"}
    topic_sentiment_map = {}

    try:
        topic_model = BERTopic(language="english", min_topic_size=3, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            topics, _ = topic_model.fit_transform(docs)
            # Generate clean names from the model
            info = topic_model.get_topic_info()
            topic_names = dict(zip(info.Topic, info.Name))
    except Exception as e:
        st.warning(f"Topic modeling degraded: {e}") # Non-breaking warning

    # 3. Batch Process Sentiment (The Performance Win)
    results = finbert(docs, padding=True, truncation=True, max_length=512, batch_size=8)
    
    pos_count = 0
    neg_count = 0
    neu_count = 0
    total_raw_score = 0.0

    # 4. Single Unified Loop
    for i, (segment, res) in enumerate(zip(mapped_segments, results)):
        label = res['label']
        score = res['score']
        raw_score = score if label == "positive" else (-score if label == "negative" else 0.0)
        
        # Update segment with all metadata
        segment.update({
            "sentiment": label,
            "sentiment_score": score,
            "raw_sentiment_score": raw_score,
            "word_count": word_counts[i],
            "topic_id": topics[i],
            "topic_name": topic_names.get(topics[i], "General/Contextual")
        })

        # Global Counters
        if label == "positive": pos_count += 1
        elif label == "negative": neg_count += 1
        else: neu_count += 1
        total_raw_score += raw_score

        # Aggregate for Topic Decomposition
        t_name = segment["topic_name"]
        if t_name not in topic_sentiment_map:
            topic_sentiment_map[t_name] = []
        topic_sentiment_map[t_name].append(raw_score)

    # 5. Advanced Metrics Calculation
    total = len(mapped_segments)
    
    # Weighted Management Confidence
    mgt_segs = [s for s in mapped_segments if s["role"] in ["CEO", "CFO", "Executive"]]
    if mgt_segs:
        total_words = sum(s["word_count"] for s in mgt_segs)
        weighted_sum = sum(s["raw_sentiment_score"] * s["word_count"] for s in mgt_segs)
        mgt_avg = weighted_sum / total_words if total_words > 0 else 0
        management_confidence = round(((mgt_avg + 1) / 2) * 100, 1)
    else:
        management_confidence = 50.0

    # Narrative Shift Index
    prep_topics = set(s["topic_id"] for s in mapped_segments if s.get("type") == "prepared" and s["topic_id"] != -1)
    qa_topics = set(s["topic_id"] for s in mapped_segments if s.get("type") == "qa" and s["topic_id"] != -1)
    
    union = prep_topics.union(qa_topics)
    jaccard_sim = len(prep_topics.intersection(qa_topics)) / len(union) if union else 1.0
    narrative_shift_index = round((1 - jaccard_sim) * 100, 1)

    # Role-based Metrics
    management_roles = ["CEO", "CFO", "Executive"]
    role_scores = {}
    for role in management_roles:
        r_segs = [s for s in mapped_segments if s.get("role") == role]
        if r_segs:
            r_avg = sum(s["raw_sentiment_score"] for s in r_segs) / len(r_segs)
            role_scores[role] = round(((r_avg + 1) / 2) * 100, 1)
        else:
            role_scores[role] = 50.0

    # Q&A Stress Indicator
    qa_mgt = [s for s in mapped_segments if s.get("type") == "qa" and s["role"] in management_roles]
    qa_stress = round((sum(1 for s in qa_mgt if s["sentiment"] == "negative") / len(qa_mgt) * 100), 1) if qa_mgt else 0.0

    # Topic Decomposition formatting
    topic_decomp = {name: round(sum(scores)/len(scores), 2) for name, scores in topic_sentiment_map.items()}

    return {
        "positive": pos_count, "negative": neg_count, "neutral": neu_count, "total": total,
        "net_score": total_raw_score / total if total > 0 else 0.0,
        "management_confidence": management_confidence,
        "individual_management_confidence": role_scores,
        "narrative_shift_index": narrative_shift_index,
        "qa_stress_indicator": qa_stress,
        "topic_decomposition": topic_decomp
    }, mapped_segments
