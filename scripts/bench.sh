#!/usr/bin/env bash
set -euo pipefail

# Full k6 load test against the contest topology. Assumes nothing is running:
# brings the stack up, waits for it, runs k6, prints the score, tears it down.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

docker compose up -d --build
trap 'docker compose down --remove-orphans' EXIT

echo "waiting for http://localhost:9999/ready ..."
for _ in $(seq 1 60); do
    curl -fsS http://localhost:9999/ready >/dev/null 2>&1 && break
    sleep 1
done
curl -fsS http://localhost:9999/ready >/dev/null

# Run the official k6 test directly. The contest's `--profile test` compose sets
# storage_opt that only works on xfs+pquota, so we invoke k6 ourselves. Mount the
# rinha repo root as /w so k6's script-relative open('./test-data.json') and the
# cwd-relative 'test/results.json' both resolve; run as root to write results.json.
rm -f "$ROOT/rinha-de-backend-2026/test/results.json"
docker run --rm --user root --network host \
    -v "$ROOT/rinha-de-backend-2026:/w" -w /w \
    grafana/k6:latest run /w/test/test.js

results="$ROOT/rinha-de-backend-2026/test/results.json"
if [ -f "$results" ]; then
    echo "=== score ==="
    python3 - "$results" <<'PY' 2>/dev/null || cat "$results"
import json, sys
d = json.load(open(sys.argv[1])); s = d["scoring"]; b = s["breakdown"]
print(f"p99={d['p99']}  final_score={s['final_score']}")
print(f"FP={b['false_positive_detections']} FN={b['false_negative_detections']} "
      f"http_errors={b['http_errors']}  failure_rate={s['failure_rate']}")
print(f"p99_score={s['p99_score']['value']}  detection_score={s['detection_score']['value']}")
PY
fi
