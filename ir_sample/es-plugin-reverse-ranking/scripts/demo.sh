#!/usr/bin/env bash
set -euo pipefail

ES_URL="${ES_URL:-http://localhost:9200}"
INDEX_NAME="reverse-ranking-response-demo"

command -v curl >/dev/null 2>&1 || {
  printf 'curl is required to run demo.sh\n' >&2
  exit 1
}

command -v python3 >/dev/null 2>&1 || {
  printf 'python3 is required to run demo.sh\n' >&2
  exit 1
}

TMP_DIR=$(mktemp -d)

curl_json() {
  local method=$1
  local path=$2
  local content_type=${3:-application/json}

  curl --silent --show-error --fail-with-body \
    -X "$method" \
    -H "Content-Type: $content_type" \
    "$ES_URL$path" \
    --data-binary @-
}

delete_index_if_present() {
  local response_file
  local status_code

  response_file="$TMP_DIR/delete-index-response.json"
  status_code=$(curl --silent --show-error \
    --output "$response_file" \
    --write-out '%{http_code}' \
    -X DELETE \
    "$ES_URL/$INDEX_NAME")

  if [[ $status_code == "200" || $status_code == "404" ]]; then
    return 0
  fi

  cat "$response_file" >&2
  printf 'unexpected status while deleting index %s: %s\n' "$INDEX_NAME" "$status_code" >&2
  return 1
}

cleanup() {
  delete_index_if_present >/dev/null 2>&1 || true
  rm -rf "$TMP_DIR"
}

trap cleanup EXIT

delete_index_if_present

curl_json PUT "/$INDEX_NAME" <<'JSON' >"$TMP_DIR/create.json"
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "properties": {
      "rank": {
        "type": "integer"
      }
    }
  }
}
JSON

curl_json POST "/$INDEX_NAME/_bulk?refresh=true" application/x-ndjson <<'JSON' >"$TMP_DIR/bulk.json"
{"index":{"_id":"0"}}
{"rank":0}
{"index":{"_id":"1"}}
{"rank":1}
{"index":{"_id":"2"}}
{"rank":2}
{"index":{"_id":"3"}}
{"rank":3}
{"index":{"_id":"4"}}
{"rank":4}
JSON

curl_json POST "/$INDEX_NAME/_search" <<'JSON' >"$TMP_DIR/normal-search.json"
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
  }
}
JSON

curl_json POST "/$INDEX_NAME/_search" <<'JSON' >"$TMP_DIR/reverse-search.json"
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
JSON

export DEMO_TMP_DIR="$TMP_DIR"

python3 <<'PY'
import json
import os
import sys


def load(name: str):
    with open(os.path.join(os.environ["DEMO_TMP_DIR"], name), encoding="utf-8") as handle:
        return json.load(handle)


def simplify_hits(payload: dict):
    return [
        {
            "_id": hit["_id"],
            "_score": hit["_score"],
            "_source": hit["_source"],
        }
        for hit in payload["hits"]["hits"]
    ]


create_response = load("create.json")
bulk_response = load("bulk.json")
normal_response = load("normal-search.json")
reverse_response = load("reverse-search.json")

if create_response.get("acknowledged") is not True:
    raise SystemExit("index creation was not acknowledged")

if bulk_response.get("errors") is not False:
    raise SystemExit("bulk indexing reported errors")

print("=== Normal search ===")
print(json.dumps(simplify_hits(normal_response), indent=2))
print("=== Reverse ranking search ===")
print(json.dumps(simplify_hits(reverse_response), indent=2))
PY