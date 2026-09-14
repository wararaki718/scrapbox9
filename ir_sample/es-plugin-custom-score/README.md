# Elasticsearch Custom Score Plugin

Elasticsearch の元の検索スコアを100倍する、固定スコアスクリプトのサンプルです。

- Elasticsearch: 8.19.21
- Java: 21
- スクリプト言語: `custom_score`
- スクリプトソース: `multiply_by_100`

Elasticsearch プラグインは本体とバージョンを完全に一致させる必要があります。このプロジェクトでは `.env` の `ELASTICSEARCH_VERSION` を Gradle と Docker Compose の両方から参照します。

## 必要なソフトウェア

- Docker と Docker Compose
- ローカルで Gradle ビルドする場合のみ Java 21

macOS で Java 21 を選択する例:

```sh
export JAVA_HOME=$(/usr/libexec/java_home -v 21)
```

## ビルド

プラグインのテストを実行し、インストール用 ZIP を作成します。

```sh
make build
```

生成物:

```text
build/distributions/custom-score-1.0.0.zip
```

Gradle を直接使う場合:

```sh
./gradlew clean test pluginZip
```

## Elasticsearch の起動

Docker 内でプラグインをビルドし、Elasticsearch にインストールして起動します。ローカルの Java は使いません。

```sh
make up
```

Elasticsearch が healthy になるまでコマンドは待機します。起動後は `http://localhost:9200` でアクセスできます。

プラグインがロードされたことを確認します。

```sh
curl -fsS 'http://localhost:9200/_cat/plugins?h=component,version'
```

期待される出力:

```text
custom-score 1.0.0
```

## スコアの確認

サンプル用インデックスを作成します。

```sh
curl -fsS -X PUT 'http://localhost:9200/custom-score-demo' \
  -H 'Content-Type: application/json' \
  -d '{
    "mappings": {
      "properties": {
        "text": { "type": "text" }
      }
    }
  }'
```

2件の文書を登録し、すぐ検索できるよう refresh します。

```sh
curl -fsS -X POST 'http://localhost:9200/_bulk?refresh=true' \
  -H 'Content-Type: application/x-ndjson' \
  --data-binary $'{"index":{"_index":"custom-score-demo","_id":"1"}}\n{"text":"elasticsearch plugin score"}\n{"index":{"_index":"custom-score-demo","_id":"2"}}\n{"text":"elasticsearch custom plugin"}\n'
```

通常の `match` クエリを実行します。

```sh
curl -fsS \
  'http://localhost:9200/custom-score-demo/_search?filter_path=hits.hits._id,hits.hits._score' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "match": {
        "text": "elasticsearch plugin"
      }
    }
  }'
```

同じクエリを `script_score` で包み、カスタムスクリプトを実行します。

```sh
curl -fsS \
  'http://localhost:9200/custom-score-demo/_search?filter_path=hits.hits._id,hits.hits._score' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "script_score": {
        "query": {
          "match": {
            "text": "elasticsearch plugin"
          }
        },
        "script": {
          "lang": "custom_score",
          "source": "multiply_by_100"
        }
      }
    }
  }'
```

各文書の `_score` が通常検索の100倍になります。例えば通常スコアが `0.36464313` の場合、カスタムスコアは丸め誤差を含む `36.464314` です。

## ログと終了

ログを表示します。

```sh
make logs
```

コンテナとデータボリュームを削除します。

```sh
make down
```

## 実装の制約

このサンプルは `ScoreScript.CONTEXT` の `multiply_by_100` だけを受け付けます。未知のスクリプトソースや別のスクリプトコンテキストは `IllegalArgumentException` で拒否します。任意の倍率や任意コードの評価には対応していません。
