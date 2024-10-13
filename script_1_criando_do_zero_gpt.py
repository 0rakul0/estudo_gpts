#%% imports
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% tokenização
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

#%% config
config = GPT2Config()
model = GPT2LMHeadModel(config)

#%% texto dos dados
with open('dados/bible.txt','r') as file:
    bible_text = file.read()

#%% separação de pedaços
chunk_size = 1024
pedacos = [bible_text[i:i + chunk_size] for i in range(0, len(bible_text), chunk_size)]

#%% tokens dos pedaços
tokens = [tokenizer(chunk, return_tensors='pt', max_length=chunk_size, truncation=True, padding='max_length') for chunk in pedacos]

#%% enviando os tokens para a gpu
tokens = [{'input_ids': t['input_ids'].to(device), 'attention_mask': t['attention_mask'].to(device)} for t in tokens]

#%% instanciando o modelo
model = model.to(device)
model.train()

#%% otimização
optmizer = torch.optim.Adam(model.parameters(), lr=2e-5)

#%% regularização do modelo caso o patience não melhorar ele para o treino
patience = 2
best_loss = float('inf')
petience_counter = 0

#%% treinando o modelo a T4 leva 30 min e a 3050 também está na media de 30 min
epochs = 5
total_steps = epochs * len(tokens)
step = 0

for epoch in range(epochs):
    total_loss = 0
    start_time = time.time()
    with tqdm(total=len(tokens), desc=f'Epoch {epoch+1}/{epochs}') as pbar:
        for batch in tokens:
            step += 1
            optmizer.zero_grad()

            outputs = model(**batch, labels=batch['input_ids'])
            loss = outputs.loss
            loss.backward()

            optmizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)
    avg_loss = total_loss / len(tokens)
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f'Epoch {epoch+1}/{epochs} completed in: {epoch_time:.2f} seconds. Average loss: {avg_loss:.4f} ')

    if avg_loss < best_loss:
        best_loss = avg_loss
        petience_counter = 0
    else:
        petience_counter += 1
    if petience_counter >= patience:
        print("Early stopping!")
        break

#%% usando modelo
model.eval()

#%% salvando o modelo
model.save_pretrained('models/bible_gpt2')
tokenizer.save_pretrained('models/bible_gpt2')

#%% fazendo perguntas ao modelo
testes = "who led the Isralites out of Egypt?"

input_ids = tokenizer.encode(testes, return_tensors='pt').to(device)
outputs = model.generate(input_ids, max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f'Q: {testes}\n R: {response}\n')
