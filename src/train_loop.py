import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
from typing import Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCS_DIR = Path("results")
DOCS_DIR.mkdir(parents=True, exist_ok=True)
CONF_MATRIX_PATH = DOCS_DIR / "confusion_matrix.png"

def train_and_evaluate(
        model, 
        X_train, 
        X_test, 
        y_train,
        y_test
) -> object:
    """
    Обучает модель и выводит метрики.

    Args:
        model (Any): модель для обучения.
        X_train, X_test, y_train, y_test: выборки данных.
    """

    # Если модель — это GridSearchCV, берём лучшую подмодель
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_

    logger.info("Training started...")

    eval_set = [(X_train, y_train), (X_test, y_test)]

    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    logger.info("Training finished.")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {acc:.4f}")

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print("CV accuracy:", np.mean(scores))

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)

    return model


def plot_confusion_matrix(
        cm: Any,
        save_conf_path: Optional[Path] = CONF_MATRIX_PATH
) -> None:
    """
    Строит confusion matrix.

    Args:
        cm (Any): confusion matrix.
    
    """
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Предсказано")
    plt.ylabel("Истинное значение")
    plt.title("Матрица ошибок")
    plt.show()
    plt.savefig(save_conf_path, dpi=200)
    plt.close()
    logger.info("Saved confusion matrix to %s", save_conf_path)



