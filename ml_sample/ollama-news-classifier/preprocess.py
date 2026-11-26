from sklearn.utils import Bunch


def preprocess(data: Bunch, n_samples: int=20) -> tuple[list[str], list[str]]:
    valid_samples = [
        (t, y) for t, y in zip(data.data, data.target) if t.strip()
    ][:n_samples]
    y_true = [data.target_names[y] for _, y in valid_samples]
    X = [t for t, _ in valid_samples]
    return X, y_true
