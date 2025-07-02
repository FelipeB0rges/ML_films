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
    # Remove HTML tags
    text = re.sub(r'<br\s*/?>', ' ', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Carregar todos os dados (treino e teste)
train_texts_raw, train_labels = load_imdb_data('aclImdb/train')
test_texts_raw, test_labels = load_imdb_data('aclImdb/test')

# Combinar dados de treino e teste em um único conjunto
all_texts_raw = train_texts_raw + test_texts_raw
all_labels = np.array(train_labels + test_labels)

# Pré-processar todos os textos
all_texts = [preprocess_text(text) for text in all_texts_raw]

# Dividir os dados em conjuntos de treino e teste estratificados
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

# Criar o vetorizador TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

# Aprender o vocabulário e transformar os dados de treino
X_train = vectorizer.fit_transform(X_train_texts)

# Transformar os dados de teste
X_test = vectorizer.transform(X_test_texts)

# Criar e treinar o modelo de Regressão Logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcular e imprimir a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo de análise de sentimentos: {accuracy:.4f}")

# Exemplo de predição com uma nova crítica
new_review = "This movie was fantastic! The acting was superb and the plot was thrilling."
new_review_processed = preprocess_text(new_review)
new_review_vectorized = vectorizer.transform([new_review_processed])
prediction = model.predict(new_review_vectorized)

print(f"\nA nova crítica: '{new_review}'")
print(f"Predição: {'Positiva' if prediction[0] == 1 else 'Negativa'}")

new_review_2 = "A complete waste of time. The plot was boring and the acting was terrible."
new_review_2_processed = preprocess_text(new_review_2)
new_review_2_vectorized = vectorizer.transform([new_review_2_processed])
prediction_2 = model.predict(new_review_2_vectorized)

print(f"\nA nova crítica: '{new_review_2}'")
print(f"Predição: {'Positiva' if prediction_2[0] == 1 else 'Negativa'}")
