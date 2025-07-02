import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- Parâmetros ---
IMG_SIZE = (128, 128)
NUM_EPOCHS = 5
BATCH_SIZE = 32

# --- Carregamento e Pré-processamento dos Dados ---

# 1. Carregar o CSV
df = pd.read_csv('MovieGenre.csv', encoding='ISO-8859-1')

# 2. Filtrar o dataframe para incluir apenas os pôsteres que foram baixados
poster_files = [f.split('.')[0] for f in os.listdir('posters')]
df['imdbId'] = df['imdbId'].astype(str)
df_filtered = df[df['imdbId'].isin(poster_files)].copy()

# 3. Processar os gêneros (dividir a string e pegar os 10 mais comuns)
df_filtered['Genre'] = df_filtered['Genre'].apply(lambda x: x.split('|'))
all_genres = [genre for sublist in df_filtered['Genre'] for genre in sublist]
genre_counts = pd.Series(all_genres).value_counts()
top_genres = genre_counts.head(10).index.tolist()

def filter_top_genres(genres):
    return [genre for genre in genres if genre in top_genres]

df_filtered['Genre_filtered'] = df_filtered['Genre'].apply(filter_top_genres)

# Remover linhas que ficaram sem nenhum gênero após a filtragem
df_filtered = df_filtered[df_filtered['Genre_filtered'].apply(len) > 0]

print(f"Número de filmes após filtragem: {len(df_filtered)}")
print(f"Top 10 gêneros: {top_genres}")

# 4. Carregar e pré-processar as imagens
def load_and_preprocess_image(imdb_id):
    try:
        img_path = os.path.join('posters', f"{imdb_id}.jpg")
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        return img_array / 255.0  # Normalizar para [0, 1]
    except Exception as e:
        # print(f"Erro ao carregar a imagem {imdb_id}: {e}")
        return None

X_images = []
y_labels = []

for index, row in tqdm(df_filtered.iterrows(), total=df_filtered.shape[0]):
    img = load_and_preprocess_image(row['imdbId'])
    if img is not None:
        X_images.append(img)
        y_labels.append(row['Genre_filtered'])

X_images = np.array(X_images)

# 5. Binarizar os labels (MultiLabelBinarizer)
mlb = MultiLabelBinarizer(classes=top_genres)
y_encoded = mlb.fit_transform(y_labels)

# --- Construção e Treinamento do Modelo ---

# 1. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_images, y_encoded, test_size=0.2, random_state=42)

# 2. Construir o modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(top_genres), activation='sigmoid') # Sigmoid para classificação multi-label
])

# 3. Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy', # Loss para multi-label
              metrics=['accuracy'])

model.summary()

# 4. Treinar o modelo
print("\nIniciando o treinamento do modelo...")
history = model.fit(X_train, y_train,
                    epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test))

# --- Avaliação ---
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nAcurácia do modelo de classificação de pôsteres: {accuracy:.4f}")

# --- Exemplo de Previsão ---

# Pegar um pôster aleatório do conjunto de teste para o exemplo
idx = np.random.randint(0, len(X_test))
poster_exemplo = np.expand_dims(X_test[idx], axis=0)
predicao_exemplo = model.predict(poster_exemplo)[0]

# Obter os gêneros previstos (com probabilidade > 0.5)
predicted_genres = mlb.classes_[(predicao_exemplo > 0.5)]
actual_genres = mlb.classes_[(y_test[idx] == 1)]

print(f"\n--- Exemplo de Previsão de Gênero por Pôster ---")
print(f"Gêneros Reais: {list(actual_genres)}")
print(f"Gêneros Previstos: {list(predicted_genres)}")
