
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
import os
import re

# Criar diretório para salvar os gráficos
if not os.path.exists('graficos'):
    os.makedirs('graficos')

# --- Gráficos para Análise de Sentimentos ---

print("Gerando gráficos para Análise de Sentimentos...")

def load_imdb_data(data_dir):
    texts, labels = [], []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(data_dir, label_type)
        for fname in sorted(os.listdir(dir_name)):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(1 if label_type == 'pos' else 0)
    return texts, labels

# Carregar dados
train_texts, train_labels = load_imdb_data('aclImdb/train')
test_texts, test_labels = load_imdb_data('aclImdb/test')
all_labels = np.array(train_labels + test_labels)

# 1. Gráfico de Distribuição das Classes
plt.figure(figsize=(6, 4))
sns.countplot(x=all_labels)
plt.title('Distribuição das Críticas (0: Negativa, 1: Positiva)')
plt.xlabel('Sentimento')
plt.ylabel('Contagem')
plt.xticks([0, 1], ['Negativa', 'Positiva'])
plt.savefig('graficos/sentimentos_distribuicao.png')
plt.close()

# 2. Nuvens de Palavras
def preprocess_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text

all_texts_raw = train_texts + test_texts
all_texts_processed = [preprocess_text(text) for text in all_texts_raw]

positive_texts = " ".join([all_texts_processed[i] for i, label in enumerate(all_labels) if label == 1])
negative_texts = " ".join([all_texts_processed[i] for i, label in enumerate(all_labels) if label == 0])

wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_texts)
wordcloud_neg = WordCloud(width=800, height=400, background_color='black').generate(negative_texts)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title('Palavras Mais Comuns em Críticas Positivas')
plt.savefig('graficos/sentimentos_wordcloud_pos.png')
plt.close()

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Palavras Mais Comuns em Críticas Negativas')
plt.savefig('graficos/sentimentos_wordcloud_neg.png')
plt.close()

# --- Gráficos para Previsão de Bilheteria ---

print("Gerando gráficos para Previsão de Bilheteria...")

df = pd.read_csv('tmdb_5000_movies.csv')
features = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'revenue']
df_clean = df[features].dropna()
df_clean = df_clean[(df_clean['budget'] > 1000) & (df_clean['revenue'] > 1000)]

# 1. Matriz de Correlação
plt.figure(figsize=(8, 6))
sns.heatmap(df_clean.corr(), annot=True, cmap='viridis', fmt='.2f')
plt.title('Matriz de Correlação das Features')
plt.savefig('graficos/bilheteria_correlacao.png')
plt.close()

# 2. Histogramas das Features
df_clean[['budget', 'popularity', 'revenue']].hist(bins=30, figsize=(12, 6))
plt.suptitle('Distribuição das Principais Features Numéricas')
plt.savefig('graficos/bilheteria_histogramas.png')
plt.close()

# --- Gráficos para Classificação de Pôsteres ---

print("Gerando gráficos para Classificação de Pôsteres...")

df_posters = pd.read_csv('MovieGenre.csv', encoding='ISO-8859-1')
poster_files = [f for f in os.listdir('posters') if f.endswith('.jpg')]
df_posters['imdbId'] = df_posters['imdbId'].astype(str)
df_filtered = df_posters[df_posters['imdbId'].isin([f.split('.')[0] for f in poster_files])].copy()

df_filtered['Genre'] = df_filtered['Genre'].apply(lambda x: x.split('|'))
all_genres = [genre for sublist in df_filtered['Genre'] for genre in sublist]
genre_counts = pd.Series(all_genres).value_counts().head(10)

# 1. Distribuição dos Gêneros
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.title('Distribuição dos 10 Gêneros Mais Comuns')
plt.xlabel('Gênero')
plt.ylabel('Contagem')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('graficos/posters_distribuicao_generos.png')
plt.close()

print("Geração de gráficos concluída!")
