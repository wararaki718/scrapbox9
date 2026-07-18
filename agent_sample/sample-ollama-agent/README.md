# LangChain + Local Ollama Quickstart

LangChain からローカルで動作する Ollama モデルに質問する最小サンプルです。既定では軽量な `qwen2.5:0.5b` を使用します。

## 前提条件

Ollama をインストールし、モデルをダウンロードします。

```bash
ollama pull qwen2.5:0.5b
```

通常、Ollama は自動的に起動します。応答できない場合は、別のターミナルで次を実行してください。

```bash
ollama serve
```

## セットアップ

リポジトリのルートからサンプルのディレクトリへ移動し、仮想環境と依存関係を準備します。

```bash
cd agent_sample/sample-ollama-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 実行

質問を一度だけ実行するには、次のように渡します。

```bash
python main.py "日本の首都はどこですか？"
```

対話モードを起動するには、引数なしで実行します。

```bash
python main.py
```

対話モードは `exit`、`quit`、`Ctrl-D`、または `Ctrl-C` で終了できます。

## モデルの変更

`OLLAMA_MODEL` 環境変数で利用するモデルを切り替えられます。たとえば `tinyllama` を使う前に、モデルをダウンロードしてください。

```bash
ollama pull tinyllama
OLLAMA_MODEL=tinyllama python main.py "こんにちは"
```

## テスト

サンプルのディレクトリで次を実行します。

```bash
python3 -m unittest discover -s tests -v
```
