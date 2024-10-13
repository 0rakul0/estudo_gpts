import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


#%% usando o modelo
model_name = './fine_tuned_gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#%% função para testar o gerador de texto
def generate_text(prompt, model, tokenizer, max_length=50):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

#%% testando
prompt = "Once upon a time"
gen = generate_text(prompt, model, tokenizer)
print(gen)

