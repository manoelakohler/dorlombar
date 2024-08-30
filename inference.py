""" Estamos em produção
1. Carregar o modelo treinado
2. Carregar o dataset a ser inferido
3. Executar a inferência
4. Apresentar os resultados
"""

# 1. Carregar o modelo treinado
import joblib
model = joblib.load('model/spine_model.pkl')

# 2. Carregar o dataset a ser inferido
import pandas as pd
data = pd.read_csv('data/Dataset_spine_unknown.csv')

# 3. Executar a inferência
inferences = model.predict(data)

# 4. Apresentar os resultados
print(inferences)
data['previsoes'] = inferences
data.to_csv('model/inferences.csv', index=False)