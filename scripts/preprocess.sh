#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESOURCE_DIR="$ROOT/data/resources"
MODEL_DIR="$ROOT/data/model"
REF_GZ="${1:-$RESOURCE_DIR/references.json.gz}"
REF_JSON="$RESOURCE_DIR/references.json"
OUT="${2:-$MODEL_DIR/references.i16.bin}"

if [[ ! -f "$REF_GZ" ]]; then
    "$ROOT/scripts/sync-contest-data.sh"
fi

mkdir -p "$MODEL_DIR"
gzip -dc "$REF_GZ" > "$REF_JSON"
zig build --release=fast preprocess -- "$REF_JSON" "$OUT"
rm -f "$REF_JSON"
