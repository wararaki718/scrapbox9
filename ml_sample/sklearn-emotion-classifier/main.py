from fetch import fetch_emotion_dataset
from preprocess import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier


def main() -> None:
    # fetch the emotion dataset
    dataset = fetch_emotion_dataset()

    # preprocess
    X_train, Y_train = preprocess(dataset["train"])
    X_validation, Y_validation = preprocess(dataset["validation"])
    print("X_train:", len(X_train))
    print("Y_train:", len(Y_train))
    print("X_validation:", len(X_validation))
    print("Y_validation:", len(Y_validation))

    # vectorize the text data
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_validation_vec = vectorizer.transform(X_validation)

    # train the classifier
    classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    classifier.fit(X_train_vec, Y_train)

    # evaluate the classifier
    Y_pred = classifier.predict(X_validation_vec)
    print(classification_report(Y_validation, Y_pred))
    print("DONE")


if __name__ == "__main__":
    main()
