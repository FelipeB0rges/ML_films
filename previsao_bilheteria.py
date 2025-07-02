import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Carregar o dataset
df = pd.read_csv('tmdb_5000_movies.csv')

# --- Pré-processamento e Feature Engineering ---

# 1. Selecionar features relevantes
features = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']
df_selected = df[features + ['revenue']].copy()

# 2. Tratar valores ausentes (preencher com a mediana)
for col in features:
    median_val = df_selected[col].median()
    df_selected.loc[:, col] = df_selected[col].fillna(median_val)

# 3. Remover filmes com orçamento ou receita zerados (não são úteis para o cálculo de ROI)
df_selected = df_selected[(df_selected['budget'] > 0) & (df_selected['revenue'] > 0)]

# 4. Definir a variável alvo (target): Sucesso de Bilheteria
# ROI (Return on Investment) = revenue / budget
# Definimos "sucesso" como um filme com ROI >= 3 (receita 3x maior que o orçamento)
df_selected['roi'] = df_selected['revenue'] / df_selected['budget']
df_selected['success'] = (df_selected['roi'] >= 3).astype(int)

# --- Treinamento do Modelo ---

# 1. Separar features (X) e target (y)
X = df_selected[features]
y = df_selected['success']

# 2. Dividir em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Treinar o modelo RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Avaliação ---

# 1. Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# 2. Calcular e imprimir a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo de previsão de sucesso: {accuracy:.4f}")

# --- Exemplo de Previsão ---

# Criar um filme hipotético para teste
filme_exemplo = pd.DataFrame({
    'budget': [20000000],       # 20 milhões
    'popularity': [80],          # Alta popularidade
    'runtime': [120],            # 120 minutos
    'vote_average': [7.5],       # Nota média boa
    'vote_count': [3000]          # Muitos votos
})

# Fazer a predição
predicao_exemplo = model.predict(filme_exemplo[features])
probabilidade_exemplo = model.predict_proba(filme_exemplo[features])

print(f"\n--- Previsão para Filme de Exemplo ---")
print(f"Características: {filme_exemplo.to_dict('records')[0]}")
print(f"Predição: {'Sucesso' if predicao_exemplo[0] == 1 else 'Fracasso'}")
print(f"Probabilidade de Sucesso: {probabilidade_exemplo[0][1]:.2%}")
