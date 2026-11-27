from datasets import Dataset


def preprocess(dataset: Dataset, columns: list[str]) -> tuple[list[str], list[list[int]]]:
    texts = dataset["Tweet"]
    Y = [[int(row[column]) for column in columns] for row in dataset]
    return texts, Y
