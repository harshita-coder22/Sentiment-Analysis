from advance_rule_based_sentiment import AdvancedRuleBasedSentimentAnalyzer

rule_model = AdvancedRuleBasedSentimentAnalyzer()
y_pred_rule_based = []

for text in test_texts:
    sentences, results = rule_model.analyze_paragraph(text)
    pred_label_str = rule_model.aggregate_sentiments(sentences, results)
    pred_label_int = label_map[pred_label_str]
    y_pred_rule_based.append(pred_label_int)
