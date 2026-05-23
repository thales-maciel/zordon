#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PORT="${PORT:-8080}"
export ZORDON_MODEL_PATH="${ZORDON_MODEL_PATH:-$ROOT/data/model/references.i16.bin}"
exec zig build run
