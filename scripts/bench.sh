#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

docker compose up -d --build
trap 'docker compose down --remove-orphans' EXIT

for _ in $(seq 1 60); do
    if curl -fsS http://localhost:9999/ready >/dev/null; then
        break
    fi
    sleep 1
done

curl -fsS http://localhost:9999/ready >/dev/null
docker compose -f "$ROOT/rinha-de-backend-2026/test/docker-compose.yml" --profile test up --abort-on-container-exit --remove-orphans
