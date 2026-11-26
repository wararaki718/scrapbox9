import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch


def fetch_news(categories: list[str]) -> Bunch:
    removes = (
        'headers',  # ヘッダー情報を削除
        'footers',  # フッター情報を削除
        'quotes'    # 引用部分を削除
    )
    test = fetch_20newsgroups(
        subset='test',
        categories=categories,
        remove=removes,
        random_state=42
    )
    return test
