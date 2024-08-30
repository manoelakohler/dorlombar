import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File

# cria a api
app = FastAPI(docs_url='/', title='Deploy DM BIMaster PUC-Rio')

# carregar o modelo treinado
model = joblib.load('model/spine_model.pkl')


# rota para inferência
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """Endpoint para inferência de dor lombar"""

    data = pd.read_csv(file.file)
    inferences = model.predict(data)
    return {'inferences': inferences.tolist()}
