import logging
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from typing import Optional

logger = logging.getLogger(__name__)

def build_model(
    objective: str = "binary:logistic",
    eval_metric: str = "logloss",
    colsample_bytree: float = 0.8,
    learning_rate: float = 0.1,
    max_depth: int = 5,
    n_estimators: int = 200,
    random_state: int = 42,
    subsample: float = 0.8,
    scaling_pos_weight: Optional[float] = 1.0,
    tune: Optional[bool] = False
) -> BaseEstimator:
    """
    Создает и возвращает модель XGBoost.

    Аргументы:
        objective (str): цель оптимизации.
        eval_metric (str): метрика для оценки.
        colsample_bytree (float): доля признаков для каждой итерации.
        learning_rate (float): скорость обучения.
        max_depth (int): максимальная глубина дерева.
        n_estimators (int): количество деревьев.
        random_state (int): сид для воспроизводимости.
        subsample (float): доля выборки для каждой итерации.
        scaling_pos_weight (Optional[float]): вес для положительного класса.
        tune (Optional[bool]): если True, выполняется GridSearchCV для тюнинга гиперпараметров.

    Выход:
        BaseEstimator: инициализированная модель.
    """
    base_model = XGBClassifier(
        objective=objective,
        eval_metric=eval_metric,
        colsample_bytree=colsample_bytree,
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=random_state,
        subsample=subsample,
        scale_pos_weight=scaling_pos_weight
    )

    if not tune:
        logger.info("XGBoost model initialized with default tuned parameters.")
        return base_model

    # --- Параметры для тюнинга ---
    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [200, 300, 400],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.8, 1.0],
    }

    logger.info("Starting GridSearchCV for XGBoost hyperparameter tuning...")

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        random_state=random_state,
        n_iter=30,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    return search
