y_pred_deep = []

for text in test_texts:
    
    pred_label_str = predict_sentiment(model, tokenizer, text)
    pred_label_int = label_map[pred_label_str]
    y_pred_deep.append(pred_label_int)

