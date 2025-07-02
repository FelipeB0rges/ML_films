import pandas as pd
import requests
import os
from tqdm import tqdm

# Criar o diretório para salvar os pôsteres, se não existir
if not os.path.exists('posters'):
    os.makedirs('posters')

# Carregar o arquivo CSV
df = pd.read_csv('MovieGenre.csv', encoding='ISO-8859-1')

# Limitar o número de pôsteres para baixar (para um teste rápido)
num_posters_to_download = 500
df_sample = df.head(num_posters_to_download)

print(f"Baixando {num_posters_to_download} pôsteres...")

# Iterar sobre o DataFrame e baixar cada pôster
for index, row in tqdm(df_sample.iterrows(), total=df_sample.shape[0]):
    poster_url = row['Poster']
    imdb_id = row['imdbId']
    
    # Definir o nome do arquivo de saída
    output_path = os.path.join('posters', f"{imdb_id}.jpg")

    # Verificar se o arquivo já existe para não baixar novamente
    if os.path.exists(output_path):
        continue

    try:
        # Fazer o download da imagem
        response = requests.get(poster_url, timeout=10)
        response.raise_for_status()  # Lança um erro para códigos de status ruins (4xx ou 5xx)

        # Salvar a imagem no arquivo
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar o pôster para o filme {imdb_id}: {e}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado para o filme {imdb_id}: {e}")

print("\nDownload de pôsteres concluído!")
