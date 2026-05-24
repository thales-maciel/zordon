# Zordon

Zig backend for Rinha de Backend 2026 fraud detection.

The upstream contest repository is checked out at `rinha-de-backend-2026/` only as local reference material. It is ignored by this repo.

## Current Shape

- `GET /ready` and `POST /fraud-score` are implemented in Zig.
- `docker-compose.yml` runs nginx on `9999` with two API instances.
- `deploy/submission/` contains the runtime-only files expected on the contest `submission` branch.
- `tools/preprocess.zig` converts the official references JSON into a compact fixed-point model.

The server starts with a tiny fallback development model if `data/model/references.i16.bin` is missing. That keeps smoke tests usable before preprocessing the full dataset, but it is not a competitive model.

## Local Flow

```sh
make test
make build
```

Or directly:

```sh
zig build test
zig build run
```

In another shell:

```sh
curl -fsS http://localhost:8080/ready
```

To build the full model from the contest checkout:

```sh
make sync-data
make preprocess
```

To run the contest topology locally:

```sh
make compose-up
```

## Shipping

The inferred public image and source repo are:

- `ghcr.io/thales-maciel/zordon:latest`
- `https://github.com/thales-maciel/zordon`

Build the amd64 image:

```sh
make image
```

Refresh the runtime-only branch:

```sh
make submission
```
