from xgboost import XGBClassifier
from sklearn.base import BaseEstimator


def build_model(
    objective: str = "binary:logistic",
    learning_rate: float = 0.1,
    n_estimators: int = 200,
    max_depth: int = 5,
    random_state: int = 42,
    eval_metric: str = "logloss"
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
        objective=objective,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        eval_metric=eval_metric,
        use_label_encoder=False
    )
