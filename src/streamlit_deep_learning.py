import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer (only loads once at app startup)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("sentiment_model")
    model = AutoModelForSequenceClassification.from_pretrained("sentiment_model")
    return tokenizer, model

tokenizer, model = load_model()

label_map = {0:'negative', 1:'neutral', 2:'positive'}

def predict_sentiment(text):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        output = model(**encoding)
        pred = torch.argmax(output.logits, dim=1).item()
    return label_map[pred]

# Streamlit interface
st.title("Deep Learning Sentiment Analysis (BERT)")
user_input = st.text_area("Paste a paragraph or sentence here:")

if st.button("Analyze"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text above.")
