import nltk
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class AdvancedRuleBasedSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.contradict_markers = [
            "but", "however", "although", "though", "yet", "despite", "on the other hand",
            "unfortunately", "nevertheless"
        ]

    def analyze_sentence(self, sentence):
        scores = self.analyzer.polarity_scores(sentence)
        compound = scores['compound']
        if compound > 0.05:
            return "positive", compound
        elif compound < -0.05:
            return "negative", compound
        else:
            return "neutral", compound

    def contains_contradiction(self, text):
        return any(marker in text.lower() for marker in self.contradict_markers)

    def analyze_paragraph(self, text):
        sentences = sent_tokenize(text)
        results = []
        for s in sentences:
            label, score = self.analyze_sentence(s)
            results.append({'sentence': s, 'sentiment': label, 'score': score})
        return sentences, results

    def aggregate_sentiments(self, sentences, sentence_results):
        if not sentence_results:
            return "neutral"
        # If contradiction marker present, weight last sentence higher
        full_text = " ".join(sentences).lower()
        if self.contains_contradiction(full_text):
            # Put more weight on last sentence or clause after "but"/marker
            # Example: weight last sentence 2x others
            weights = [1]*len(sentence_results)
            weights[-1] = 2
            avg = sum(w * s['score'] for w, s in zip(weights, sentence_results)) / sum(weights)
        else:
            avg = sum(s['score'] for s in sentence_results) / len(sentence_results)
        if avg > 0.05:
            return "positive"
        elif avg < -0.05:
            return "negative"
        else:
            return "neutral"
