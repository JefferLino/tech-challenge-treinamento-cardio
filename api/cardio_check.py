from fastapi import FastAPI, HTTPException
from pydantic  import BaseModel, Field
import joblib
import pandas as pd
import os

# =========================
# Config
# =========================
MODEL_PATH = os.getenv("MODEL_PATH", "pipe_logistic_regression.pkl")
THRESHOLD = 0.399676365687596

# Criando os campos do servico REST API
class CardioDataRequest(BaseModel):
        age: int = Field(..., description="idade"), 
        gender: int = Field(..., description="genero (1 - Feminino, 2 - Masculino)"), 
        height: int = Field(..., description="altura(cm)"), 
        weight: int = Field(..., description="peso(kg)"), 
        ap_hi: int = Field(..., description="pressão arterial sistólica"), 
        ap_lo: int = Field(..., description="pressão arterial diastólica"), 
        cholesterol: int = Field(..., description="cholesterol (1- normal, 2- acima do normal, 3- alto)"),
        gluc: int = Field(..., description="glicose (1- normal, 2- acima do normal, 3- alto)"), 
        smoke: int = Field(..., description="fumante (1- Sim, 0- Não)"), 
        alco: int = Field(..., description="alcólatra (1- Sim, 0- Não)"), 
        active: int = Field(..., description="exercicios fisicos (1- Sim, 0- Não)")

# Carregando o nosso pipeline treinado
try: 
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    pipeline = None
    load_error = str(e)

app = FastAPI(title="FIAP - TechChallenge - Cardio Check", version="1.0")

class PredictResponse(BaseModel):
    prob_cardio_decease: float
    threshold: float
    result: int  # 1 risco de doenca, 0 sem risco


@app.post("/predict", response_model=PredictResponse)
def predict(req: CardioDataRequest):

    X = pd.DataFrame([{
        "age": req.age,
        "gender": req.gender,
        "height": req.height,
        "weight": req.weight,
        "ap_hi": req.ap_hi,
        "ap_lo": req.ap_lo,
        "cholesterol": req.cholesterol,
        "gluc": req.gluc,
        "smoke": req.smoke,
        "alco": req.alco,
        "active": req.active
    }])

    print(X)
 
    # Predição de probabilidade (classe 1 = risco)
    try:
        proba = float(pipeline.predict_proba(X)[:, 1])
        pred_1 = float(pipeline.predict(X))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    pred = 1 if proba >= THRESHOLD else 0

    return PredictResponse(
        prob_cardio_decease=proba,
        threshold=THRESHOLD, 
        result=pred
    )
