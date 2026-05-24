#!/usr/bin/env bash
set -euo pipefail

# Quick k6 smoke test against the contest topology. Assumes nothing is running:
# brings the stack up, waits for it, runs the smoke checks, tears it down.
# Exits non-zero if any k6 threshold fails.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

docker compose up -d --build
trap 'docker compose down --remove-orphans' EXIT

echo "waiting for http://localhost:9999/ready ..."
for _ in $(seq 1 30); do
    curl -fsS http://localhost:9999/ready >/dev/null 2>&1 && break
    sleep 1
done
curl -fsS http://localhost:9999/ready >/dev/null

# Run k6 directly (the contest's `--profile smoke` compose sets storage_opt that
# only works on xfs+pquota). Mount the rinha repo root as /w; run as root.
docker run --rm --user root --network host \
    -v "$ROOT/rinha-de-backend-2026:/w" -w /w \
    grafana/k6:latest run /w/test/smoke.js
