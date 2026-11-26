from sklearn.metrics import classification_report
from tqdm import tqdm

from classify import classify
from fetch import fetch_news
from preprocess import preprocess


def main() -> None:
    categories = [
        "sci.space",          # 宇宙科学
        "rec.autos",          # 自動車
        "comp.graphics",      # コンピュータグラフィックス
        "talk.politics.misc", # 政治
    ]
    # Fetch and preprocess the test data
    test_data = fetch_news(categories)
    X_test, y_true = preprocess(test_data, n_samples=5)

    # Classify the test data and print the classification report
    y_pred = []
    for text in tqdm(X_test):
        result = classify(text, categories)
        y_pred.append(result.category)
    print(classification_report(y_true, y_pred, zero_division=0))
    print("DONE")


if __name__ == "__main__":
    main()
