# Matryoshka Representation Learning

日本語のクエリ・正例文ペアを PyTorch の Matryoshka 表現学習で学習する自己完結サンプルです。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make run
python3 -m unittest discover -s tests -v
```

外部データは `{"query": "質問", "positive": "関連文書"}` 形式の JSONL として用意し、`make run ARGS="--data-path pairs.jsonl"` を指定します。`--embedding-dim`、`--dimensions`、`--epochs`、`--batch-size`、`--device` で学習設定を変更できます。
