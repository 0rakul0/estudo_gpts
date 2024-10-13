"""
O objetivo desse script é usar um modelo BERT com LoRA
"""

# %% imports
import torch
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch.nn as nn
import pandas as pd

# %% Verifique se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Usando o dispositivo: {device}')

# %% Import do dataset
dataset = load_dataset('imdb')

# %% Pegando a parte de treino (convertendo para DataFrame para visualização)
df_train = pd.DataFrame(dataset['train'])

# %% Importando o modelo
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=2)

# %% Definindo LoRA
class LoRA(nn.Module):
    def __init__(self, input_dim, rank=4):
        super(LoRA, self).__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(input_dim, rank))
        self.B = nn.Parameter(torch.randn(rank, input_dim))

    def forward(self, x):
        return x + torch.matmul(torch.matmul(x, self.A), self.B)

# %% Bert com o LoRA
class BertWithLoRA(nn.Module):
    def __init__(self, model, lora_rank=4):
        super(BertWithLoRA, self).__init__()
        self.bert = model.bert
        self.lora = LoRA(self.bert.config.hidden_size, rank=lora_rank)
        self.classifier = model.classifier

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state
        adapted_states = self.lora(hidden_states)
        logits = self.classifier(adapted_states[:, 0, :])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return {"loss": loss, "logits": logits}

# Instanciando o modelo com LoRA
model_with_lora = BertWithLoRA(model)

# %% Função de preprocessamento
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length")

# Tokenizando o dataset
tokenizer_datasets = dataset.map(preprocess_function, batched=True)
tokenizer_datasets = tokenizer_datasets.rename_column("label", "labels")
tokenizer_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# %% Argumentos de treinamento
training_args = TrainingArguments(
    output_dir='./results',  # Diretório de saída
    evaluation_strategy="epoch",  # Avaliação por época
    learning_rate=2e-5,  # Taxa de aprendizado
    per_device_train_batch_size=8,  # Tamanho do lote
    per_device_eval_batch_size=8,  # Tamanho do lote para validação
    num_train_epochs=2,  # Número de épocas
    weight_decay=0.01,  # Decaimento de peso
)

# %% Trainer
trainer = Trainer(
    model=model_with_lora,
    args=training_args,
    train_dataset=tokenizer_datasets['train'],
    eval_dataset=tokenizer_datasets['test'],
)

# %% Treinamento uma T4 leva 1 hora, a 3050 de 8gb leva 3 horas
trainer.train()


#%% salvando o modelo
torch.save(model_with_lora.state_dict(), "./results/model_with_lora.pth")

results = trainer.evaluate()
print(results)