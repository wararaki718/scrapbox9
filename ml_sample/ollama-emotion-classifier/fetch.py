from datasets import load_dataset, DatasetDict


def fetch_emotion_dataset(
    path: str = "SemEvalWorkshop/sem_eval_2018_task_1",
) -> DatasetDict:
    return load_dataset(path, revision="refs/convert/parquet")
