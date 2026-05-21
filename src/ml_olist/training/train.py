import os
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.ml_olist.training.data import load_olist_data, split_data, validate_schema
from src.ml_olist.training.features import build_preprocessing_pipeline

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model_latest.joblib"


def train(n_estimators: int = 150, max_depth: int = 15, random_state: int = 42) -> dict:
    """Train RandomForestRegressor pour prédire delivery_time_days."""

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "olist-delivery-time"))

    with mlflow.start_run() as run:
        print(f"MLflow run_id : {run.info.run_id}")

        df = load_olist_data()
        validate_schema(df)

        X_train, X_test, y_train, y_test = split_data(df)

        print(f"Train : {len(X_train)} lignes | Test : {len(X_test)} lignes")

        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        })

        model = Pipeline([
            ("preprocessing", build_preprocessing_pipeline()),
            ("regressor", RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )),
        ])

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        }

        mlflow.log_metrics(metrics)

        print(f"R²={metrics['r2']:.4f} MAE={metrics['mae']:.1f} RMSE={metrics['rmse']:.1f}")

        # ⚠️ Gate qualité
        if metrics["r2"] < 0.70:
            warnings.warn(f"R² faible: {metrics['r2']:.4f}")

        # ✅ LOCAL SAVE ONLY
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print(f"Model saved locally → {MODEL_PATH}")

    return metrics


if __name__ == "__main__":
    train()

