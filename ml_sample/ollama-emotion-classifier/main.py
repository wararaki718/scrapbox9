from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from classify import classify
from fetch import fetch_emotion_dataset
from preprocess import preprocess


def main() -> None:
    emotion_columns = [
        "anger", "anticipation", "disgust", "fear", "joy",
        "love", "optimism", "pessimism", "sadness", "surprise", "trust"
    ]

    # fetch and preprocess the validation dataset
    dataset = fetch_emotion_dataset()
    X_validations, Y = preprocess(
        dataset["validation"].select(range(30)),
        emotion_columns,
    )

    # classify each text in the validation set
    predictions = []
    for text in tqdm(X_validations):
        predictions.append(classify(text, emotion_columns))

    # postprocess
    mlb = MultiLabelBinarizer(classes=emotion_columns)
    Y_preds = mlb.fit_transform(predictions) 

    # evaluate
    print(classification_report(Y, Y_preds, target_names=emotion_columns, zero_division=0))

    print("DONE")


if __name__ == "__main__":
    main()
