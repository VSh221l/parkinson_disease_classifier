from src.data import load_data, preprocess_data
from src.model import build_model
from src.train_loop import train_and_evaluate
from src.utils import save_model


def main():
    print("=== Обнаружение болезни Паркинсона ===")

    # 1. Загружаем данные
    df = load_data()

    # 2. Предобработка

    X_train, X_test, y_train, y_test = preprocess_data(df, normalize=True)

    # 3. Строим модель
    model = build_model()

    # 4. Обучение и оценка
    trained_model = train_and_evaluate(model, X_train, X_test, y_train, y_test)

    # 5. Сохраняем модель
    save_model(trained_model)


if __name__ == "__main__":
    main()