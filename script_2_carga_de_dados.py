#%% imports
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from datasets import load_dataset
import pandas as pd

#%% Verifique se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Usando o dispositivo: {device}')

#%% gerando modelos
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Mova o modelo para o dispositivo (GPU se disponível)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
small_dataset = dataset.shuffle(seed=42).select(range(1000))

#%% carregando a base
df = pd.DataFrame(small_dataset)
print(df.head(10))

#%% tokenizando
def tokenize_func(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)

# Transforma em um json de informações
tokenize_dataset = small_dataset.map(tokenize_func, batched=True, remove_columns=['text'])

#%% o dataframe
df = pd.DataFrame(tokenize_dataset)
print(df.head(10))

#%% preparando os dados para o modelo
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

#%% carregando para o modelo
batch_size = 4
data_loader = DataLoader(tokenize_dataset, batch_size=batch_size, collate_fn=data_collator)

# Certifique-se de que os dados também vão para o dispositivo
for batch in data_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    break

for key, value in batch.items():
    print(f'key: {key}, value: {value}')

#%% parametros de treino
treino_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_eval_batch_size=batch_size,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=50,
)

#%% criando o treinador
trainer = Trainer(
    model=model,
    args=treino_args,
    train_dataset=tokenize_dataset,
    data_collator=data_collator
)

#%% treinando o modelo
trainer.train()

#%% salvando o modelo
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained('./fine_tuned_gpt2')
