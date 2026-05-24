#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-ghcr.io/thales-maciel/zordon:latest}"

cd "$ROOT"
docker buildx build --platform linux/amd64 -t "$IMAGE" .
