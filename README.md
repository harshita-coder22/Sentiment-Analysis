Sentiment Analysis: Rule-Based and Deep Learning Models
Overview
This project automates the detection of sentiments from paragraphs and predicts the overall sentiment. It combines a traditional rule-based approach enhanced with contradiction detection alongside a modern transformer-based deep learning model using DistilBERT. The goal is to compare, evaluate, and deploy an efficient, accurate sentiment analysis system.

Features
Advanced Rule-Based Model: Built on VADER with added contradiction detection and weighted sentence aggregation for mixed sentiment handling.

Deep Learning Model: Fine-tuned DistilBERT transformer model with custom dataset classes and training pipeline for high accuracy.

Evaluation: Comprehensive metrics including accuracy, precision, recall, F1-score; confusion matrices and classification reports.

Interactive Web App: Streamlit application for real-time sentiment analysis of user input text with confidence scoring.

Modular Code: Clean, extensible structure for easy customization and further development.

Getting Started
Prerequisites
Python 3.8 or above

Recommended to use a virtual environment

Installation
bash
git clone https://github.com/harshita-coder22/Sentiment-Analysis.git
cd Sentiment-Analysis
python -m venv sentiment_env
source sentiment_env/bin/activate   # Linux/Mac
.\sentiment_env\Scripts\activate    # Windows
pip install -r requirements.txt
Running the Streamlit App
bash
streamlit run src/sentiment_comparison_app.py
Open the URL provided by Streamlit to interact with the app.

Training the Deep Learning Model
bash
python src/train_model.py
Adjust parameters as needed in the training script.

Evaluation
To evaluate models on test data and compare:

bash
python src/evaluate_models.py
Project Structure
text
sentiment_analysis/
├── src/
│   ├── advance_rule_based_sentiment.py
│   ├── training_utils.py
│   ├── evaluate_models.py
│   ├── sentiment_comparison_app.py
│   ├── train_model.py
│   └── ...
├── data/
│   └── IMDB_Dataset.csv
├── sentiment_model/
│   ├── config.json
│   ├── tokenizer files
│   └── model.safetensors
├── requirements.txt
└── README.md
Contribution
Feel free to submit issues or pull requests for improvements or bug fixes.
