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
    label_map_inv = {0:'negative', 1:'neutral', 2:'positive'}
    return label_map_inv[pred]

sample_text = "I did not enjoy the movie but the soundtrack was good."
print(predict_sentiment(sample_text))
