#!/usr/bin/env bash
set -euo pipefail

# Build the runtime-only `submission` branch the contest expects: docker-compose.yml
# at the root (which pulls the public GHCR image), info.json, LICENSE and a short
# README — and NO source code. Regenerated from main each run. Run after the image
# is published to GHCR.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE="$ROOT/.submission-worktree"

cd "$ROOT"
git worktree remove --force "$WORKTREE" 2>/dev/null || true
git worktree add -B submission "$WORKTREE" main

# Wipe the working tree (keep .git) and lay down only the runtime files.
find "$WORKTREE" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp deploy/submission/docker-compose.yml "$WORKTREE/docker-compose.yml"
cp deploy/submission/info.json "$WORKTREE/info.json"
cp LICENSE "$WORKTREE/LICENSE"
cat >"$WORKTREE/README.md" <<'MD'
# Zordon — submission branch

Runtime-only branch for Rinha de Backend 2026. `docker compose up` pulls the
public image `ghcr.io/thales-maciel/zordon:latest` — a custom load balancer plus
two API instances over a shared Unix-socket volume — and serves on port 9999.

Source code, build and benchmarks: https://github.com/thales-maciel/zordon (`main`).
MD

(
    cd "$WORKTREE"
    git add -A
    git commit --quiet -m "submission: runtime docker-compose + info.json"
    echo "submission branch built — files:"
    git ls-files
)

cat <<EOF

Ready. Publish with:
  git push -f origin submission

(The image must be published & public first: push main so the build-image
workflow runs, then make the GHCR package public.)
EOF
