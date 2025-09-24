import nltk
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class RuleBasedSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentence(self, sentence):
        scores = self.analyzer.polarity_scores(sentence)
        compound = scores['compound']
        if compound > 0.05:
            return "positive", compound
        elif compound < -0.05:
            return "negative", compound
        else:
            return "neutral", compound

    def analyze_paragraph(self, text):
        sentences = sent_tokenize(text)
        results = []
        for s in sentences:
            label, score = self.analyze_sentence(s)
            results.append({'sentence': s, 'sentiment': label, 'score': score})
        return results

    def aggregate_sentiments(self, sentence_results):
        if not sentence_results:
            return "neutral"
        avg = sum(s['score'] for s in sentence_results) / len(sentence_results)
        if avg > 0.05:
            return "positive"
        elif avg < -0.05:
            return "negative"
        else:
            return "neutral"
