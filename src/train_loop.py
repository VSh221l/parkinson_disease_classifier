import logging
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
from typing import Any, Optional, Tuple

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
) -> Tuple[object, float]:
    """
    Обучает модель и выводит метрики.

    Args:
        model (Any): модель для обучения.
        X_train, X_test, y_train, y_test: выборки данных.
    """
    logger.info("Training started...")
    model.fit(X_train, y_train)
    logger.info("Training finished.")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {acc:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)

    return model, acc


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



