import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_URL = "https://storage.yandexcloud.net/academy.ai/practica/parkinsons.data"
DEFAULT_LOCAL_PATH = Path("data/parkinsons.csv")

def load_data(
        path_or_url: Optional[str] = DEFAULT_URL, 
        save_to: Optional[str] = DEFAULT_LOCAL_PATH
) -> pd.DataFrame:
    """
    Загружает CSV с локального пути или по URL.
    - Если path_or_url не указан → используется DEFAULT_URL.
    - Файл сохраняется локально (если задан save_to).
    Делит данные на признаки (X) и метки (y).
    """
    try:
        log.info("Loading dataset from %s", path_or_url)

        df = pd.read_csv(path_or_url)

        if save_to:
            save_to.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_to, index=False)
            log.info("Saved dataset to %s", save_to)

        return df
    
    except Exception as e:
        log.exception("Failed to load dataset: %s", e)
        raise


def preprocess_data(
    df: pd.DataFrame, normalize: bool = True, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Делит данные на train/test и выполняет нормализацию (опционально).

    Аргументы:
    -   df (pd.DataFrame): исходный датасет.
    -   normalize (bool): нужно ли нормализовать признаки.
    -   test_size (float): доля тестовой выборки.
    -   random_state (int): сид для воспроизводимости.
    """
    if "name" in df.columns:
            df = df.drop(columns=["name"])  # убираем строковые имена
    
    X = df.drop(columns=["status"])
    y = df["status"]

    non_numeric_cols = X.select_dtypes(include=["object"]).columns
    if len(non_numeric_cols) > 0:
        log.warning("Removed non-numeric columns: %s", list(non_numeric_cols))
        X = X.drop(columns=non_numeric_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    """
    Для XGBoost нормализация не обязательна, но мы применяем её ради консистентности и расширяемости пайплайна.
    """
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        log.info("Data normalized with StandardScaler.")
    else:
        log.info("Data used without normalization.")

    return X_train, X_test, y_train, y_test