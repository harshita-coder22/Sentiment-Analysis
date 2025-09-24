import streamlit as st
import nltk
from transformers import pipeline
from advance_rule_based_sentiment import AdvancedRuleBasedSentimentAnalyzer

nltk.download('punkt')

@st.cache_resource
def load_rule_model():
    return AdvancedRuleBasedSentimentAnalyzer()

@st.cache_resource
def load_dl_pipeline():
    return pipeline("sentiment-analysis", model="../sentiment_model", tokenizer="../sentiment_model")

rule_model = load_rule_model()
dl_pipeline = load_dl_pipeline()

def get_rule_based_sentiment_and_score(text):
    sentences, results = rule_model.analyze_paragraph(text)
    pred_label_str = rule_model.aggregate_sentiments(sentences, results)
    # Calculate confidence as average absolute compound score
    scores = [abs(r['score']) for r in results]
    confidence = sum(scores) / len(scores) if scores else 0.0
    if pred_label_str == "neutral":
        pred_label_str = "negative"  # Treat neutral as negative for binary consistency
    return pred_label_str, confidence

def get_dl_sentiment_and_score(text, max_len=512):
    safe_text = text[:max_len]
    result = dl_pipeline(safe_text)[0]
    label = "negative" if result['label'] == 'LABEL_0' else "positive"
    score = result['score']  # Model confidence score (probability)
    return label, score

st.title("Sentiment Analysis with Scores: Rule-Based vs Deep Learning")

user_input = st.text_area("Enter text to analyze sentiment:")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text above.")
    else:
        with st.spinner("Analyzing..."):
            rb_label, rb_score = get_rule_based_sentiment_and_score(user_input)
            dl_label, dl_score = get_dl_sentiment_and_score(user_input)
        
        st.write("### Results")
        st.write(f"**Rule-Based Model Sentiment:** {rb_label} (Confidence: {rb_score:.3f})")
        st.write(f"**Deep Learning Model Sentiment:** {dl_label} (Confidence: {dl_score:.3f})")
