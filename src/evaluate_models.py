import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from transformers import pipeline
import nltk

from advance_rule_based_sentiment import AdvancedRuleBasedSentimentAnalyzer

# --- Load and Prepare Data ---
try:
    df = pd.read_csv("../data/IMDB_Dataset.csv")
except FileNotFoundError:
    print("Error: IMDB_Dataset.csv not found. Make sure it's in a 'data' folder in your project directory.")
    exit()

# Use a smaller sample for time
df_test = df.sample(n=200, random_state=42)
test_texts = df_test['review'].tolist()
true_labels_str = df_test['sentiment'].tolist()
label_map = {"negative": 0, "positive": 1}
true_labels_int = [label_map[label] for label in true_labels_str]

# --- Test the Rule-Based Model ---
print("Running evaluation for Rule-Based Model...")
rule_analyzer = AdvancedRuleBasedSentimentAnalyzer()
y_pred_rule_based = []
for text in test_texts:
    sentences, results = rule_analyzer.analyze_paragraph(text)
    pred_label_str = rule_analyzer.aggregate_sentiments(sentences, results)
    if pred_label_str == "neutral":
        pred_label_int = 0
    else:
        pred_label_int = label_map[pred_label_str]
    y_pred_rule_based.append(pred_label_int)

# --- TEST THE DEEP LEARNING MODEL (Safe Truncation) ---
print("Running evaluation for Deep Learning Model...")
try:
    dl_pipeline = pipeline("sentiment-analysis", model="../sentiment_model", tokenizer="../sentiment_model")
except Exception as e:
    print(f"Error: Deep learning model not found or failed to load. Ensure 'sentiment_model' folder exists and has model files.")
    print(f"Details: {e}")
    exit()

# Robust truncation
MAX_LEN = 512
safe_test_texts = [text[:MAX_LEN] for text in test_texts]
dl_results = dl_pipeline(safe_test_texts)
y_pred_dl = [1 if result['label'] == 'LABEL_1' else 0 for result in dl_results]

# --- Print Metrics ---
print("\n--- Rule-Based Model Metrics ---")
print("Accuracy:", accuracy_score(true_labels_int, y_pred_rule_based))
print("Precision:", precision_score(true_labels_int, y_pred_rule_based, average='weighted'))
print("Recall:", recall_score(true_labels_int, y_pred_rule_based, average='weighted'))
print("F1-score:", f1_score(true_labels_int, y_pred_rule_based, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(true_labels_int, y_pred_rule_based))
print("Report:\n", classification_report(true_labels_int, y_pred_rule_based))

print("\n--- Deep Learning Model Metrics ---")
print("Accuracy:", accuracy_score(true_labels_int, y_pred_dl))
print("Precision:", precision_score(true_labels_int, y_pred_dl, average='weighted'))
print("Recall:", recall_score(true_labels_int, y_pred_dl, average='weighted'))
print("F1-score:", f1_score(true_labels_int, y_pred_dl, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(true_labels_int, y_pred_dl))
print("Report:\n", classification_report(true_labels_int, y_pred_dl))
