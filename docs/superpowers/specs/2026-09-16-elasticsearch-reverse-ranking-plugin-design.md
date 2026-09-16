# Elasticsearch Reverse Ranking Plugin Design

## Goal

Elasticsearch の元検索で得た上位 N 件を候補として固定し、その候補のスコアを逆数へ置換して降順に並び替えるプラグインを作成する。動作確認を自動化するスクリプトも提供する。

対象バージョンは Elasticsearch 8.19.21、Java 21 とする。候補集合を全体上位 N 件として厳密に扱うため、対象インデックスは 1 shard を前提とする。

## Ranking Semantics

元スコアを `score` とすると、再スコアは次のように定義する。

```text
score > 0 なら 1 / score
score == 0 なら 0
```

正の元スコアが低い文書ほど高い再スコアを得る。元スコア 0 は末尾に置く。負値または非有限値は不正な入力として検索を失敗させる。

呼び出し側は `_search` の `size` と `rescore.window_size` に同じ N を指定する。これにより、元検索の上位 N 件だけを返却候補として逆順化する。

```json
{
  "size": 3,
  "query": {
    "match_all": {}
  },
  "rescore": {
    "window_size": 3,
    "reverse_rank": {}
  }
}
```

## Architecture

プラグインは Elasticsearch の `SearchPlugin.RescorerSpec` 拡張点を使い、`reverse_rank` rescorer を登録する。

### ReverseRankingPlugin

Elasticsearch に `reverse_rank` の parser と transport reader を登録するエントリーポイント。

### ReverseRankingRescorerBuilder

標準 rescore 句の `window_size` を受け取り、rescore context を構築する。`reverse_rank` の設定オブジェクトは空とし、未知フィールドを拒否する。transport serialization、XContent 出力、等価性を実装する。

### ReverseRankingRescorer

shard の `TopDocs` に含まれる先頭 `window_size` 件へ逆数変換を適用し、変換後スコアの降順に再ソートする。`explain` は元スコア、変換式、変換後スコアを示す。

## Data Flow

1. Elasticsearch が通常の query を評価し、元スコア順の候補を収集する。
2. `ReverseRankingRescorerBuilder` が `window_size` を含む context を生成する。
3. `ReverseRankingRescorer` が候補の先頭 N 件を逆数スコアへ置換する。
4. Elasticsearch が変換後スコアの降順で候補を返す。

1 shard では `size == window_size == N` によって元検索の全体上位 N 件だけが候補になる。複数 shard では標準 rescore と同様に各 shard の上位 N 件が処理対象になるため、本設計の保証範囲外とする。

## Build And Runtime

隣接する `es-plugin-custom-score` と構成を揃える。

- Gradle 8.10.2
- Elasticsearch 8.19.21
- Java 21
- plugin ZIP: `build/distributions/reverse-ranking-1.0.0.zip`
- plugin name: `reverse-ranking`
- Docker Compose によるプラグイン組み込み済み Elasticsearch の起動

Make ターゲットは `build`、`up`、`verify`、`logs`、`down` を提供する。

## Error Handling

- `window_size <= 0` はリクエストエラーにする。
- `reverse_rank` 内の未知フィールドは parse error にする。
- 負値または NaN、Infinity の元スコアは明示的な例外にする。
- Elasticsearch とプラグインのバージョン不一致は plugin descriptor によりインストール時に拒否する。

## Testing

Java 単体テストで次を確認する。

- 正のスコアが逆数へ変換される。
- 0 が 0 のままになる。
- 先頭 N 件だけが変換対象になる。
- 変換後スコアの降順に並ぶ。
- 不正スコアを拒否する。
- builder の parse、serialization、XContent、等価性が成立する。
- `explain` が元スコアと逆数変換を表す。

`scripts/verify.sh` は 1 shard の一時インデックスを作り、決定的な元スコア `4, 3, 2, 1` を持つ文書を登録する。N を 3 として、通常検索の ID 順が `4, 3, 2`、再ランキング後が `2, 3, 4` となり、候補外の ID `1` が含まれないことを検証する。再スコアが各元スコアのおおよその逆数であることも Python 3 の標準ライブラリで検証する。

README にはビルド、起動、プラグイン確認、`make verify`、停止の手順と 1 shard 制約を記載する。