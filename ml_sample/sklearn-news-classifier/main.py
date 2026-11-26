from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from fetch import fetch_news


def main() -> None:
    # fetch the news dataset
    train, test = fetch_news()

    # vectorize the text data
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    X_train = vectorizer.fit_transform(train.data)
    X_test = vectorizer.transform(test.data)

    # train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train.target)

    # evaluate model on test data
    y_pred = model.predict(X_test)
    print(classification_report(test.target, y_pred))
    print("DONE")


if __name__ == "__main__":
    main()
