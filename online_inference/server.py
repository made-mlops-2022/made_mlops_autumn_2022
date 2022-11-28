from fastapi import FastAPI, HTTPException, status
from schemas import Patient
import numpy as np
from model import Model


model = Model()
app = FastAPI()


@app.get('/')
def home():
    pass


@app.get('/health', status_code=200)
def model_ready():
    try:
        model.is_ready()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'model dont ready',
        )


@app.post('/predict')
def predict(item: Patient):
    X_array = np.array([item.age, item.sex, item.cp, item.trestbps, 
                        item.chol, item.fbs, item.restecg, item.thalach, 
                        item.exang, item.oldpeak, item.slope, item.ca, 
                        item.thal])

    return model.predict(X_array)
