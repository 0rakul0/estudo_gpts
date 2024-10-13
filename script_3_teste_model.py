import torch
from transformers import BertTokenizer, BertForTokenClassification
from script_3_transfer_learning import BertWithLoRA

#%% usando o modelo
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model_with_lora = BertWithLoRA(model)
model_with_lora.load_state_dict(torch.load("./results/model_with_lora.pth"))

#%% função para testar o gerador de texto
def predict_sentiment(review_text):
    inputs = tokenizer(review_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    model_with_lora.eval()
    with torch.no_grad():
        outputs = model_with_lora(**inputs)
    logits = outputs["logits"]
    predicted_class = torch.argmax(logits, dim=1).item()
    sentiment = "positive" if predicted_class == 1 else "negative"
    return sentiment

reviews = ["This movie was fantastic! the storyline was gripping and the characters were well-developed",
           "I did not enjoy this film. the plot was predictable and the acting was mediocre at best"]

for review in reviews:
    sentiment = predict_sentiment(review)