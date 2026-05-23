#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTEST_DIR="${CONTEST_DIR:-$ROOT/rinha-de-backend-2026}"
RESOURCE_DIR="$ROOT/data/resources"

mkdir -p "$RESOURCE_DIR"

cp "$CONTEST_DIR/resources/normalization.json" "$RESOURCE_DIR/normalization.json"
cp "$CONTEST_DIR/resources/mcc_risk.json" "$RESOURCE_DIR/mcc_risk.json"
cp "$CONTEST_DIR/resources/references.json.gz" "$RESOURCE_DIR/references.json.gz"

printf 'synced contest resources into %s\n' "$RESOURCE_DIR"
