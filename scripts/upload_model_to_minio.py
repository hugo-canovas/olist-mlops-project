import sys
from pathlib import Path

MODEL_PATH = Path("artifacts/model_latest.joblib")


def upload(model_path: Path = MODEL_PATH) -> None:
    if not model_path.exists():
        print(f"Model {model_path} does not exist. Run training first.")
        sys.exit(1)

    # Ici on ne fait plus d'upload cloud
    print(f"Model ready locally -> {model_path.resolve()}")


def download_model(model_path: Path = MODEL_PATH) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    return model_path


if __name__ == "__main__":
    upload()

