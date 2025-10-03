from xgboost import XGBClassifier
from sklearn.base import BaseEstimator


def build_model(
    learning_rate: float = 0.1,
    n_estimators: int = 200,
    max_depth: int = 5,
    random_state: int = 42
) -> BaseEstimator:
    """
    Создает и возвращает модель XGBoost.

    Аргументы:
        learning_rate (float): скорость обучения.
        n_estimators (int): число деревьев.
        max_depth (int): глубина дерева.
        random_state (int): сид.

    Выход:
        BaseEstimator: инициализированная модель.
    """
    return XGBClassifier(
        objective="binary:logistic",
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss"
    )
