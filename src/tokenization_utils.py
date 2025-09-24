from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')


encoded = tokenizer(
    texts[0], 
    padding='max_length', 
    truncation=True, 
    max_length=128, 
    return_tensors="pt"
)
print(encoded)
