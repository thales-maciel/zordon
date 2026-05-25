# Zordon — submission branch

Runtime-only branch for Rinha de Backend 2026. `docker compose up` pulls the
public image `ghcr.io/thales-maciel/zordon:latest` — a custom load balancer plus
two API instances over a shared Unix-socket volume — and serves on port 9999.

Source code, build and benchmarks: https://github.com/thales-maciel/zordon (`main`).
