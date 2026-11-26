import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch


def fetch_news() -> tuple[Bunch, Bunch]:
    categories = [
        "sci.space",          # 宇宙科学
        "rec.autos",          # 自動車
        "comp.graphics",      # コンピュータグラフィックス
        "talk.politics.misc", # 政治
    ]
    removes = (
        'headers',  # ヘッダー情報を削除
        'footers',  # フッター情報を削除
        'quotes'    # 引用部分を削除
    )
    train = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=removes,
        random_state=42
    )
    test = fetch_20newsgroups(
        subset='test',
        categories=categories,
        remove=removes,
        random_state=42
    )

    return train, test
