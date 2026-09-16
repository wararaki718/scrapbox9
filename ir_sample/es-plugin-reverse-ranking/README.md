# Elasticsearch Reverse Ranking Plugin

元の検索結果の上位 N 件だけを対象に、各 `_score` を逆数へ変換して降順に並べ替える rescorer プラグインのサンプルです。

- Elasticsearch: 8.19.21
- Java: 21
- rescorer 名: `reverse_rank`

Elasticsearch プラグインは本体とバージョンを完全に一致させる必要があります。このプロジェクトでは `.env` の `ELASTICSEARCH_VERSION` を Gradle と Docker Compose の両方から参照します。

## 必要なソフトウェア

- Docker と Docker Compose
- `make verify` には `curl` と Python 3 が必要
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
build/distributions/reverse-ranking-1.0.0.zip
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
reverse-ranking 1.0.0
```

## 逆順リランキングの確認

`make verify` は 1 シャードの固定データを作成し、元の上位 3 件と rescore 後の上位 3 件を比較します。

> 注意: `scripts/verify.sh` は `ES_URL` 上の固定インデックス `reverse-ranking-demo` を毎回削除して作り直し、終了時にも削除します。保持したいクラスタには `ES_URL` を向けないでください。既定の `ES_URL` は `http://localhost:9200` です。

- 元の上位 ID は `4, 3, 2`
- `reverse_rank` 適用後は `2, 3, 4`
- `rank=0` の文書は逆数を取っても `0` のままなので最後尾に残ります

実行コマンド:

```sh
make verify
```

成功時の出力:

```text
ES_URL=http://localhost:9200 ./scripts/verify.sh
reverse ranking verification passed
```

検証で使う検索リクエスト例:

```json
{
  "size": 3,
  "query": {
    "script_score": {
      "query": {
        "match_all": {}
      },
      "script": {
        "source": "doc['rank'].value"
      }
    }
  },
  "rescore": {
    "window_size": 3,
    "reverse_rank": {}
  }
}
```

この例では元スコアが `4, 3, 2` の 3 文書だけが rescore 対象です。それぞれ `1/4, 1/3, 1/2` に変換されるため、結果順は `2, 3, 4` になります。

## スコアリング仕様

- `score > 0` のとき `1 / score` を新しい `_score` に使います
- `score == 0` のとき新しい `_score` は `0` です
- 負のスコアは拒否します
- `NaN` と `Infinity` を含む非有限値は拒否します
- 逆数計算の結果がオーバーフローして非有限値になる入力も拒否します

## 重要な制約

- グローバルな top N を逆順にできるのは、`number_of_shards=1` かつ `size == window_size == N` のときだけです
- 複数シャードでは Elasticsearch の rescore は各シャードの top N に対して適用されるため、全体 top N を一度に逆順にはしません
- `size=N` の検索では、元の top N に入っていなかった候補が rescore によって新しく返ることはありません
- `window_size` は必須で、`0` 以下は拒否します
- `reverse_rank` は空オブジェクトだけを受け付けます。未知のフィールドを含めると拒否します

## ログと終了

ログを表示します。

```sh
make logs
```

コンテナとデータボリュームを削除します。

```sh
make down
```
