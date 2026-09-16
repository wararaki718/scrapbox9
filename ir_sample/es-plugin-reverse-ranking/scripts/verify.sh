#!/usr/bin/env bash
set -euo pipefail

ES_URL="${ES_URL:-http://localhost:9200}"
INDEX_NAME="reverse-ranking-demo"
TMP_DIR=$(mktemp -d)

command -v python3 >/dev/null 2>&1 || { printf 'python3 is required to run verify.sh\n' >&2; exit 1; }

cleanup() {
  delete_index_if_present >/dev/null 2>&1 || true
  rm -rf "$TMP_DIR"
  return 0
}

trap cleanup EXIT

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
  return 1
}

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
    curl --silent --show-error --fail-with-body \
      -X "$method" \
      "$ES_URL$path"
  fi
}

delete_index_if_present

curl_json PUT "/$INDEX_NAME" '{
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
}' >"$TMP_DIR/create.json"

curl_json POST "/$INDEX_NAME/_bulk?refresh=true" $'{"index":{"_id":"0"}}\n{"rank":0}\n{"index":{"_id":"1"}}\n{"rank":1}\n{"index":{"_id":"2"}}\n{"rank":2}\n{"index":{"_id":"3"}}\n{"rank":3}\n{"index":{"_id":"4"}}\n{"rank":4}\n' >"$TMP_DIR/bulk.json"

curl_json POST "/$INDEX_NAME/_search" '{
  "size": 3,
  "query": {
    "script_score": {
      "query": {
        "match_all": {}
      },
      "script": {
        "source": "doc['"'"'rank'"'"'].value"
      }
    }
  }
}' >"$TMP_DIR/base.json"

curl_json POST "/$INDEX_NAME/_search" '{
  "size": 3,
  "query": {
    "script_score": {
      "query": {
        "match_all": {}
      },
      "script": {
        "source": "doc['"'"'rank'"'"'].value"
      }
    }
  },
  "rescore": {
    "window_size": 3,
    "reverse_rank": {}
  }
}' >"$TMP_DIR/rerank.json"

curl_json POST "/$INDEX_NAME/_search" '{
  "size": 5,
  "query": {
    "script_score": {
      "query": {
        "match_all": {}
      },
      "script": {
        "source": "doc['"'"'rank'"'"'].value"
      }
    }
  },
  "rescore": {
    "window_size": 5,
    "reverse_rank": {}
  }
}' >"$TMP_DIR/zero.json"

export VERIFY_TMP_DIR="$TMP_DIR"

python3 <<'PY'
import json
import math
import os


def load(name: str):
    with open(os.path.join(os.environ["VERIFY_TMP_DIR"], name), encoding="utf-8") as handle:
        return json.load(handle)


def hit_ids(payload):
    return [hit["_id"] for hit in payload["hits"]["hits"]]


def hit_scores(payload):
    return [hit["_score"] for hit in payload["hits"]["hits"]]


create_response = load("create.json")
bulk_response = load("bulk.json")
base_response = load("base.json")
rerank_response = load("rerank.json")
zero_response = load("zero.json")

assert create_response["acknowledged"] is True
assert bulk_response["errors"] is False
assert hit_ids(base_response) == ["4", "3", "2"]
assert hit_ids(rerank_response) == ["2", "3", "4"]
for actual, expected in zip(hit_scores(rerank_response), [0.5, 1.0 / 3.0, 0.25]):
    assert math.isclose(actual, expected, rel_tol=1e-6), (actual, expected)
assert hit_ids(zero_response) == ["1", "2", "3", "4", "0"]

print("reverse ranking verification passed")
PY