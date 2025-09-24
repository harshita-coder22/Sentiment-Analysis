import streamlit as st
from advance_rule_based_sentiment import AdvancedRuleBasedSentimentAnalyzer

st.title("Sentiment Analysis Demo")
st.write("Enter text below to get its sentiment.")

# Create a text area for user input
user_input = st.text_area("Enter text for sentiment analysis:")

if user_input:
    # 1. Create an instance of your sentiment analyzer class
    analyzer = AdvancedRuleBasedSentimentAnalyzer()

    # 2. Analyze the paragraph and get the overall sentiment
    sentences, results = analyzer.analyze_paragraph(user_input)
    overall_sentiment = analyzer.aggregate_sentiments(sentences, results)

    # 3. Display the final result to the user
    st.write("---") # This creates a horizontal line for visual separation
    st.markdown(f"**Overall Sentiment:** {overall_sentiment.capitalize()}")
