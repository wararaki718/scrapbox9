# Reverse Ranking Response Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one command that inserts deterministic Elasticsearch data, runs normal and reciprocal reranked searches, and prints both responses for comparison.

**Architecture:** A standalone Bash script owns the temporary demo index lifecycle and HTTP requests. Python 3 reads the saved Elasticsearch responses and prints only `_id`, `_score`, and `_source` as formatted JSON. Make and README expose the script without mixing it into the assertion-oriented `verify.sh`.

**Tech Stack:** Bash, curl, Python 3 standard library, Elasticsearch 8.19.21, GNU Make, Docker Compose

---

## File Structure

- `ir_sample/es-plugin-reverse-ranking/scripts/demo.sh`: create data, run two searches, print responses, and clean up.
- `ir_sample/es-plugin-reverse-ranking/Makefile`: expose `make demo` using the configured `ES_URL`.
- `ir_sample/es-plugin-reverse-ranking/README.md`: document the sample command, output, and destructive index lifecycle.

### Task 1: Response Demo Script

**Files:**
- Create: `ir_sample/es-plugin-reverse-ranking/scripts/demo.sh`
- Modify: `ir_sample/es-plugin-reverse-ranking/Makefile`

- [ ] **Step 1: Verify the missing command fails**

Run:

```bash
cd ir_sample/es-plugin-reverse-ranking
make demo
```

Expected: FAIL because the `demo` target does not exist.

- [ ] **Step 2: Create the response demo script**

Create executable `scripts/demo.sh` with this implementation:

```bash
#!/usr/bin/env bash
set -euo pipefail

ES_URL="${ES_URL:-http://localhost:9200}"
INDEX_NAME="reverse-ranking-response-demo"
TMP_DIR=$(mktemp -d)

command -v curl >/dev/null 2>&1 || { printf 'curl is required to run demo.sh\n' >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { printf 'python3 is required to run demo.sh\n' >&2; exit 1; }

curl_json() {
  local method=$1
  local path=$2
  local data=${3-}

  if [[ -n $data ]]; then
    curl --silent --show-error --fail-with-body \
      -X "$method" \
      -H 'Content-Type: application/json' \
      "$ES_URL$path" \
      --data-binary @- <<<"$data"
  else
    curl --silent --show-error --fail-with-body -X "$method" "$ES_URL$path"
  fi
}

delete_index_if_present() {
  local status_code
  status_code=$(curl --silent --show-error \
    --output "$TMP_DIR/delete.json" \
    --write-out '%{http_code}' \
    -X DELETE "$ES_URL/$INDEX_NAME")
  [[ $status_code == "200" || $status_code == "404" ]]
}

cleanup() {
  delete_index_if_present >/dev/null 2>&1 || true
  rm -rf "$TMP_DIR"
}

trap cleanup EXIT
delete_index_if_present

curl_json PUT "/$INDEX_NAME" '{
  "settings": { "number_of_shards": 1, "number_of_replicas": 0 },
  "mappings": { "properties": { "rank": { "type": "integer" } } }
}' >/dev/null

curl_json POST "/$INDEX_NAME/_bulk?refresh=true" $'{"index":{"_id":"0"}}\n{"rank":0}\n{"index":{"_id":"1"}}\n{"rank":1}\n{"index":{"_id":"2"}}\n{"rank":2}\n{"index":{"_id":"3"}}\n{"rank":3}\n{"index":{"_id":"4"}}\n{"rank":4}\n' >/dev/null

QUERY='{
  "size": 3,
  "query": {
    "script_score": {
      "query": { "match_all": {} },
      "script": { "source": "doc['"'"'rank'"'"'].value" }
    }
  }
}'

RERANK_QUERY='{
  "size": 3,
  "query": {
    "script_score": {
      "query": { "match_all": {} },
      "script": { "source": "doc['"'"'rank'"'"'].value" }
    }
  },
  "rescore": {
    "window_size": 3,
    "reverse_rank": {}
  }
}'

curl_json POST "/$INDEX_NAME/_search" "$QUERY" >"$TMP_DIR/base.json"
curl_json POST "/$INDEX_NAME/_search" "$RERANK_QUERY" >"$TMP_DIR/rerank.json"

export DEMO_TMP_DIR="$TMP_DIR"
python3 <<'PY'
import json
import os


def hits(name: str):
    with open(os.path.join(os.environ["DEMO_TMP_DIR"], name), encoding="utf-8") as handle:
        payload = json.load(handle)
    return [
        {"_id": hit["_id"], "_score": hit["_score"], "_source": hit["_source"]}
        for hit in payload["hits"]["hits"]
    ]


print("=== Normal search ===")
print(json.dumps(hits("base.json"), ensure_ascii=False, indent=2))
print("=== Reverse ranking search ===")
print(json.dumps(hits("rerank.json"), ensure_ascii=False, indent=2))
PY
```

- [ ] **Step 3: Expose the Make target**

Update the declaration and add the target:

```makefile
.PHONY: build up demo verify down logs

demo:
	ES_URL=$(ES_URL) ./scripts/demo.sh
```

- [ ] **Step 4: Validate syntax and expected connection failure**

Run while Elasticsearch is stopped:

```bash
bash -n scripts/demo.sh
make demo
```

Expected: syntax check passes; `make demo` fails with a curl connection error, proving it attempts the workflow.

- [ ] **Step 5: Run against Elasticsearch and inspect output**

Run:

```bash
make up
make demo | tee /tmp/reverse-ranking-demo-output.txt
grep -F '=== Normal search ===' /tmp/reverse-ranking-demo-output.txt
grep -F '=== Reverse ranking search ===' /tmp/reverse-ranking-demo-output.txt
python3 - <<'PY'
from pathlib import Path

output = Path("/tmp/reverse-ranking-demo-output.txt").read_text()
normal, reranked = output.split("=== Reverse ranking search ===")
assert [normal.index(f'"_id": "{value}"') for value in (4, 3, 2)] == sorted(
    normal.index(f'"_id": "{value}"') for value in (4, 3, 2)
)
assert [reranked.index(f'"_id": "{value}"') for value in (2, 3, 4)] == sorted(
    reranked.index(f'"_id": "{value}"') for value in (2, 3, 4)
)
PY
curl -sS -o /dev/null -w '%{http_code}\n' \
  'http://localhost:9200/reverse-ranking-response-demo'
make down
```

Expected: both headings are present, IDs appear in the expected order, the final index check prints `404`, and Docker cleanup succeeds.

- [ ] **Step 6: Commit the executable sample**

```bash
git add ir_sample/es-plugin-reverse-ranking/scripts/demo.sh ir_sample/es-plugin-reverse-ranking/Makefile
git commit -m "feat: add reverse ranking response demo"
```

### Task 2: Demo Documentation

**Files:**
- Modify: `ir_sample/es-plugin-reverse-ranking/README.md`

- [ ] **Step 1: Document the demo workflow**

Add a section after plugin startup containing:

```markdown
## 検索レスポンスの確認

サンプルデータの登録、通常検索、逆順検索、レスポンス表示を一度に実行します。

```sh
make demo
```

通常検索は ID `4, 3, 2`、逆順検索は ID `2, 3, 4` の順で、各 hit の `_id`、`_score`、`_source` を整形済み JSON として表示します。

> 注意: `scripts/demo.sh` は `ES_URL` 上の固定インデックス `reverse-ranking-response-demo` を削除して作成し、終了時にも削除します。
```

Update the prerequisites line so both `make demo` and `make verify` require curl and Python 3.

- [ ] **Step 2: Run final verification**

```bash
make build
make up
make demo
make verify
make down
git diff --check
```

Expected: build succeeds, demo prints both JSON result sets, verification passes, cleanup succeeds, and no whitespace errors are reported.

- [ ] **Step 3: Commit documentation**

```bash
git add ir_sample/es-plugin-reverse-ranking/README.md
git commit -m "docs: document response demo"
```