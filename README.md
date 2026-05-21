# Olist — Pipeline MLOps Classification Satisfaction
![CI](https://github.com/<votre-org>/olist-mlops/actions/workflows/ci.yml/badge.svg)
## Stack
- **ML** : scikit-learn (RandomForestRegressor) + MLflow
- **Stockage artefacts** : MinIO (S3-compatible)
- **Backend** : FastAPI (Python 3.11)
- **Frontend** : Streamlit (Python)
- **CI/CD** : GitHub Actions
- **Infra locale** : Docker Compose
## Métriques de référence (holdout set 20%)
| Métrique | Valeur |
| -------- | ------ |
| R² | > 0.80 |
| MAE | < 5 jours |
| RMSE | < 7 jours |