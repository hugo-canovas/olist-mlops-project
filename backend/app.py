import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

from tests.test_data import df
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_olist.prediction.schemas import OrderFeatures, DeliveryPrediction
from backend.storage.s3_client import download_model

MODEL_LOCAL = os.getenv("LOCAL_MODEL_PATH", "/app/tmp/model_latest.joblib")
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load(download_model(MODEL_LOCAL))
    print("Modèle Olist chargé.")
    yield
    model = None
    
app = FastAPI(
    title="Olist Delivery Time API",
    description="Prédit le temps de livraison (jours) d'une commande Olist",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=DeliveryPrediction)
def predict(order: OrderFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    df = pd.DataFrame([order.model_dump()])
    pred = float(model.predict(df)[0])
    return DeliveryPrediction(delivery_time_days=pred)