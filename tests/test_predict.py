import pytest
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

MODEL_PATH = Path("artifacts/model_latest.joblib")

# Commande type : Sao paulo -> Sao paulo, pric moyen, livraison rapide
SAMPLE = pd.DataFrame([{
    "price": 89.90,
    "feight_value": 12.50,
    "payment_installments": 1,
    "payment_value": 102.40,
    "order_item_id": 1,
    "order_purchase_timestamp": datetime(2018, 6, 10, 9, 0),
    "seller_state": "SP",
    "customer_state": "SP",
}])

@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip("model absent - lancer train.py d'abord")
    import joblib
    return joblib.load(MODEL_PATH)

def test_prediction_is_positive(model):
    pred = model.predict(SAMPLE)[0]
    assert pred > 0, f"Temps predit négatif: {pred}"

def test_prediction_is_under_60_days(model):
    pred = model.predict(SAMPLE)[0]
    assert pred < 60, f"Temps prédit irréaliste: {pred.lf} jours"

def test_model_r2_threshold(model):
    """ Gate CI : R² doit dépasser 0.70 sur le holdout set."""
    from src.ml_olist.training.data import load_olist_data, split_data
    from sklearn.metrics import r2_score
    df = load_olist_data()
    _, X_te, _, Y_te = split.data(df)
    r2 = r2_score(Y_te, model.predict(X_te))
    assert r2 >= 0.70, f"R² trop faible: {r2:.4f}"