
# Relatório Final: Aplicação de Aprendizado Supervisionado na Indústria Cinematográfica

**Integrantes:** Felipe Borges e Fernando Filter

## 1. Introdução

Este relatório detalha o desenvolvimento de três aplicações de aprendizado de máquina supervisionado para resolver problemas comuns na indústria cinematográfica. O projeto abrange a classificação de textos, dados tabulares e imagens, demonstrando a versatilidade da inteligência artificial em diferentes contextos.

As três tarefas desenvolvidas foram:
1.  **Análise de Sentimentos:** Classificação de críticas de filmes como positivas ou negativas.
2.  **Previsão de Sucesso de Bilheteria:** Previsão do retorno sobre o investimento (ROI) de um filme com base em dados tabulares.
3.  **Classificação de Gênero por Pôster:** Classificação de gênero de filmes a partir de seus pôsteres.

## 2. Análise de Sentimentos (Texto)

### 2.1. Objetivo

O objetivo desta tarefa foi criar um modelo capaz de classificar críticas de filmes do dataset IMDb como "positivas" ou "negativas".

### 2.2. Metodologia

Utilizamos um modelo de **Regressão Logística** combinado com um vetorizador **TF-IDF**. O TF-IDF transforma o texto das críticas em vetores numéricos que podem ser processados pelo modelo. Para garantir a robustez do treinamento, combinamos os dados de treino e teste e os dividimos novamente de forma estratificada.

### 2.3. Resultados

O modelo alcançou uma acurácia de **89.47%**, demonstrando alta eficácia na classificação de sentimentos.

### 2.4. Código-Fonte (`analise_sentimentos.py`)

```python
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

def load_imdb_data(data_dir):
    """Carrega os dados do IMDB de um diretório específico."""
    texts = []
    labels = []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(data_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(1 if label_type == 'pos' else 0)
    return texts, labels

def preprocess_text(text):
    """Limpa o texto removendo tags HTML e caracteres não alfabéticos."""
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text

all_texts_raw = train_texts_raw + test_texts_raw
all_labels = np.array(train_labels + test_labels)

all_texts = [preprocess_text(text) for text in all_texts_raw]

X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo de análise de sentimentos: {accuracy:.4f}")
```

## 3. Previsão de Sucesso de Bilheteria (Dados Tabulares)

### 3.1. Objetivo

Prever se um filme será um "sucesso" de bilheteria, definido como um Retorno Sobre o Investimento (ROI) de pelo menos 3 (receita 3x maior que o orçamento).

### 3.2. Metodologia

Utilizamos o dataset "TMDB 5000 Movie Dataset". As features selecionadas foram `budget`, `popularity`, `runtime`, `vote_average` e `vote_count`. O modelo escolhido foi um **RandomForestClassifier**, que é robusto e bom para lidar com dados tabulares.

### 3.3. Resultados

O modelo alcançou uma acurácia de **72.45%** na previsão de sucesso.

### 3.4. Código-Fonte (`previsao_bilheteria.py`)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('tmdb_5000_movies.csv')

features = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']
df_selected = df[features + ['revenue']].copy()

for col in features:
    median_val = df_selected[col].median()
    df_selected.loc[:, col] = df_selected[col].fillna(median_val)

df_selected = df_selected[(df_selected['budget'] > 0) & (df_selected['revenue'] > 0)]

df_selected['roi'] = df_selected['revenue'] / df_selected['budget']
df_selected['success'] = (df_selected['roi'] >= 3).astype(int)

X = df_selected[features]
y = df_selected['success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo de previsão de sucesso: {accuracy:.4f}")
```

## 4. Classificação de Gênero por Pôster (Imagens)

### 4.1. Objetivo

Classificar o gênero de um filme a partir da imagem de seu pôster. Esta é uma tarefa de classificação de imagens multi-label, pois um filme pode ter vários gêneros.

### 4.2. Metodologia

Utilizamos o dataset "Movie Poster Dataset". Primeiro, fizemos o download de 500 pôsteres. Em seguida, construímos uma **Rede Neural Convolucional (CNN)** com Keras/TensorFlow. O modelo foi treinado para reconhecer os 10 gêneros mais comuns no dataset.

### 4.3. Resultados

O modelo alcançou uma acurácia de **57.53%**. Este resultado é um ponto de partida e pode ser melhorado com mais dados, mais tempo de treinamento e uma arquitetura de modelo mais complexa.

### 4.4. Código-Fonte (`classificacao_posters.py`)

```python
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

IMG_SIZE = (128, 128)
NUM_EPOCHS = 5
BATCH_SIZE = 32

df = pd.read_csv('MovieGenre.csv', encoding='ISO-8859-1')
poster_files = [f.split('.')[0] for f in os.listdir('posters')]
df['imdbId'] = df['imdbId'].astype(str)
df_filtered = df[df['imdbId'].isin(poster_files)].copy()

df_filtered['Genre'] = df_filtered['Genre'].apply(lambda x: x.split('|'))
all_genres = [genre for sublist in df_filtered['Genre'] for genre in sublist]
genre_counts = pd.Series(all_genres).value_counts()
top_genres = genre_counts.head(10).index.tolist()

def filter_top_genres(genres):
    return [genre for genre in genres if genre in top_genres]

df_filtered['Genre_filtered'] = df_filtered['Genre'].apply(filter_top_genres)
df_filtered = df_filtered[df_filtered['Genre_filtered'].apply(len) > 0]

X_images = []
y_labels = []

for index, row in df_filtered.iterrows():
    img_path = os.path.join('posters', f"{row['imdbId']}.jpg")
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    X_images.append(img_array / 255.0)
    y_labels.append(row['Genre_filtered'])

X_images = np.array(X_images)

mlb = MultiLabelBinarizer(classes=top_genres)
y_encoded = mlb.fit_transform(y_labels)

X_train, X_test, y_train, y_test = train_test_split(X_images, y_encoded, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(top_genres), activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia do modelo de classificação de pôsteres: {accuracy:.4f}")
```

## 5. Conclusão

Este projeto demonstrou com sucesso a aplicação de técnicas de aprendizado supervisionado em três domínios diferentes de dados da indústria cinematográfica. Os resultados mostram que é possível extrair informações valiosas e fazer previsões úteis a partir de textos, dados tabulares e imagens, abrindo portas para diversas aplicações práticas, desde a análise de feedback de audiência até a automação da catalogação de filmes.
