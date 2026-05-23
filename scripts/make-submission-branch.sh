#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE="$ROOT/.submission-worktree"

cd "$ROOT"
git worktree remove --force "$WORKTREE" 2>/dev/null || true
git worktree add -B submission "$WORKTREE" main

find "$WORKTREE" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp deploy/submission/docker-compose.yml "$WORKTREE/docker-compose.yml"
cp deploy/submission/nginx.conf "$WORKTREE/nginx.conf"
cp deploy/submission/info.json "$WORKTREE/info.json"

(
    cd "$WORKTREE"
    git add docker-compose.yml nginx.conf info.json
    git status --short
)

printf 'submission worktree ready at %s\n' "$WORKTREE"
