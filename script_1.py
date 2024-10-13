"""
pip install spacy
python -m spacy download pt_core_news_sm
"""

#%% imports
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
import re
from sklearn.model_selection import train_test_split
import spacy

#%% carregando o basico
nlp = spacy.load("pt_core_news_sm")
nltk.download('punkt_tab')
nltk.download('stopwords')

#%% teste de base
data = {'text': ['Nós adoramos python', 'python é legal', 'Ciência de dados é legal', None, 'Estou correndo para aprender python','eu tive aulas muito complicadas', 'gosto de aprender programação']}
df = pd.DataFrame(data)
print(df)

#%% troca o None por ''
df['text'].fillna('', inplace=True)

#%% remoçao de ruido
def remover_ruido(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]','', text)
    text = text.lower().strip()
    return text

df['texto_limpo'] = df['text'].apply(remover_ruido)

#%% texto tokenizado
df['texto_tokenizado'] = df['texto_limpo'].apply(word_tokenize)

#%% removendo stopwords
stop_words = set(stopwords.words('portuguese'))
df['sem_stopwords'] = df['texto_tokenizado'].apply(lambda x: [word for word in x if word not in stop_words])

#%% stemmer da palavra e lemmatização
stemmer = PorterStemmer()
df['stemmer_text'] = df['sem_stopwords'].apply(lambda x: [stemmer.stem(word) for word in x])

def lemmatizer(text):
    doc = nlp(" ".join(text))
    return [token.lemma_ for token in doc]

df['lemmatizer_text'] = df['sem_stopwords'].apply(lemmatizer)

#%% tokenizar ele faz uma representação matrixial das palavras
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['lemmatizer_text'])
sequences = tokenizer.texts_to_sequences(df['lemmatizer_text'])
one_hot_results = tokenizer.texts_to_matrix(df['lemmatizer_text'], mode='binary')

print(one_hot_results)

#%% mostrando a pocisão das palavras
word_index = tokenizer.word_index
print("frequencia de palavras", word_index)

#%% criando o proprio word embeddins
max_len = 10
padded_sequences = pad_sequences(sequences, maxlen=max_len)
print(padded_sequences)

#%% frequencia de palavras
word_freq = Counter([word for sublist in df['lemmatizer_text'] for word in sublist])
print('palavras comuns', word_freq.most_common(5))

#%% separando treino e teste
X_train, X_temp, y_train, y_temp = train_test_split(df['lemmatizer_text'], df['text'], test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print("conjunto de treino", len(X_train))
print("conjunto de validação", len(X_val))
print("conjunto de teste", len(X_test))