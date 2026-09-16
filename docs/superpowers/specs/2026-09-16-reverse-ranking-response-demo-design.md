# Reverse Ranking Response Demo Design

## Goal

Elasticsearch へのサンプルデータ投入、通常検索、`reverse_rank` を使った検索、レスポンス表示を一つのコマンドで実行できるようにする。

## Interface

`ir_sample/es-plugin-reverse-ranking/scripts/demo.sh` を追加し、`make demo` から実行する。接続先は既存スクリプトと同じく `ES_URL` で変更でき、既定値は `http://localhost:9200` とする。

## Data Flow

1. 固定インデックス `reverse-ranking-response-demo` が存在すれば削除する。
2. 1 shard、replica なし、integer 型の `rank` フィールドを持つインデックスを作る。
3. ID と `rank` が 0 から 4 の文書を bulk API で登録し、refresh する。
4. `doc['rank'].value` をスコアにする通常検索を `size=3` で実行する。
5. 同じ検索へ `window_size=3` と空の `reverse_rank` を加えて実行する。
6. 通常検索と逆順検索の `hits.hits` を、見出し付きの整形 JSON として順番に表示する。
7. 終了時にデモ用インデックスを削除する。

表示対象は各 hit の `_id`、`_score`、`_source` とする。通常検索では ID が `4, 3, 2`、逆順検索では `2, 3, 4` となり、スコア変換前後を比較できる。

## Error Handling

- `curl` と Python 3 がない場合は、HTTP リクエストを送る前に明確なエラーで終了する。
- Elasticsearch の HTTP エラーではレスポンス本文を標準エラーへ表示する。
- 途中で失敗しても trap で一時ディレクトリとデモ用インデックスを削除する。
- 固定インデックスを削除・再作成することを README に明記する。

## Testing

- `bash -n scripts/demo.sh` で構文を確認する。
- Elasticsearch 停止中に実行し、接続エラーになることを RED として確認する。
- プラグイン入り Elasticsearch を起動し、`make demo` が成功することを確認する。
- 出力に通常検索と逆順検索の見出し、ID 順 `4,3,2` と `2,3,4`、対応する `_score` と `_source` が含まれることを確認する。
- 実行後に `reverse-ranking-response-demo` が存在しないことを確認する。