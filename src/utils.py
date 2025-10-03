import joblib
import os


def save_model(model, path="models/xgb_parkinsons.pkl") -> None:
    """Сохраняет обученную модель."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Модель сохранена в {path}")


def load_model(path="models/xgb_parkinsons.pkl"):
    """Загружает модель из файла."""
    return joblib.load(path)