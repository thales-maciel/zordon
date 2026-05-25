# Zordon

Zig backend for the Rinha de Backend 2026 fraud-detection challenge: classify each
transaction by an exact k=5 nearest-neighbour vote over ~3M reference vectors (14
features), under ~1 CPU and ~350 MB total, on a 2014 Mac Mini (Haswell, AVX2).

Result: **p99 ≈ 1.52 ms, 0 false positives / 0 false negatives, final score ≈ 5818 (rank #19).**

The upstream contest repository is checked out at `rinha-de-backend-2026/` only as
local reference material; it is git-ignored.

## Approach

- **Exact 5-NN, no approximation.** Detection is scored on the approve/deny decision,
  and we keep it perfect (0 FP / 0 FN) — no candidate cap, no learned shortcut.
- **i16 quantization** (scale 10000); 14 features padded to 16 lanes so a squared
  distance is a single `@Vector(16, i16)` AVX2 op. Vectors are reordered into a compact
  binary model that is built at image-build time and `mmap`'d at startup.
- **Bucketed KD-tree index** (`src/model.zig`, `src/classifier.zig`). Vectors are first
  split into 16 buckets by 4 binary flags (these classes are nearly pure), then each
  bucket holds a KD-tree (median split on the widest dimension). Every node carries the
  full 16-dim bounding box of its subtree; a depth-first search prunes a whole subtree
  once the squared distance from the query to that box reaches the current 5th-nearest.
  Visiting the nearer child first lets an outlier prune almost the entire tree instead
  of scanning its (up to ~1M-vector) bucket.
- **fd-passing load balancer** (`src/lb.zig`). A tiny Zig LB accepts on `:9999` and
  hands the raw client socket to an API worker via `SCM_RIGHTS` over a Unix socket —
  zero byte copy, near-zero CPU, no proxy hop. Each API is a single-threaded epoll
  server (`src/server.zig`), one per ~0.45 CPU slice.

## What we learned

The score climbed in three measured steps, all keeping detection perfect (0 FP / 0 FN):

| change | p99 | final |
|---|---|---|
| exact scan, candidate cap, 2-D cell grid | 1453 ms | 2073 |
| full 16-dim bounding-box pruning (`model v2`) | 558 ms | 3253 |
| per-bucket KD-tree (`model v3`) | **1.52 ms** | **≈5818** |

- **Measure on the target, not the dev box.** Our dev i9 is ~5× faster per core than
  the contest Haswell; locally p99 looked sub-millisecond while the real box sat at
  1453 ms. The preview test (`rinha/test` issue) was the only ground truth — plus an
  offline sweep (`zig build sweep`) over the public test set, which reproduces exact
  detection and scan-cost numbers to iterate against without a deploy.
- **It was throughput saturation, not just a slow tail.** The load ramps to 900 req/s
  against 0.45 CPU per API — roughly a 1 ms CPU/request budget. When the *mean* query
  exceeded it, the queue diverged and p99 pinned to the 2001 ms timeout (also producing
  spurious HTTP errors). Lowering the mean made the errors vanish.
- **The pruning bound's dimensionality is everything.** A 2-D grid bound can't prune a
  cell that's near in 2 dims but far in the other 12, so outlier queries scanned the
  whole ~1M-vector bucket. A full 16-dim bounding box cut the mean 4.4×; making it
  hierarchical (KD-tree) cut the *worst-case* scan 42× (≈1M → 24k). Same exact answer.
- **A provable "decision-locked" early exit did not help.** Stopping once the fraud
  vote is mathematically settled is correct, but the slow outliers only settle after
  scanning nearly the whole bucket anyway. Measured, reverted — the win was structural.
- **`:latest` gets cached.** Re-running a preview after rebuilding `:latest` silently
  ran the *old* image. Pin the immutable commit-SHA image tag (or `pull_policy: always`)
  so the build you mean is the build that gets scored.

## Local flow

```sh
make test
make build
```

The server starts with a tiny fallback development model if
`data/model/references.i16.bin` is missing — enough for smoke tests, not competitive.
Build the real model from the contest checkout:

```sh
make sync-data
make preprocess
```

Run the contest topology (LB + two APIs) locally on `:9999`:

```sh
make compose-up
```

Offline detection + scan-cost sweep over the full public test set:

```sh
zig build sweep --release=fast
```

## Shipping

Public image and source repo:

- `ghcr.io/thales-maciel/zordon:latest`
- `https://github.com/thales-maciel/zordon`

```sh
make image        # build the amd64 image (also done in CI on push to main)
make submission   # refresh the runtime-only `submission` branch
```

The `submission` branch carries only `docker-compose.yml` + `info.json` (no source).
Pin its image to a commit-SHA tag when running a preview test, so the engine pulls the
exact build rather than a cached `:latest`.
