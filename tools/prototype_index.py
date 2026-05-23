#!/usr/bin/env python3
"""Prototype + validate the two-stage exact NN index before porting to Zig.

Stage 1: hard-partition by 4 discrete bits (online, card_present, unknown, has_history).
Stage 2: per-bucket 2-D grid over the bucket's 2 highest-variance dims, searched with
         branch-and-bound (visit cells in increasing lower-bound order, stop when the
         next cell's LB >= current 5th-best squared distance).

Everything runs in i16 quantized space (scale 10000) so the exactness result ports
directly to the Zig implementation. We validate against:
  - i16 brute force (the index MUST match this exactly), and
  - float brute force (to measure the decision flips quantization itself causes).

Query methodology: hold out a random sample of reference rows as queries; search the
full set but exclude the query's own row (test payloads are not allowed as queries).
"""
import sys
import time
import numpy as np

DIMS = 14
SCALE = 10000
K = 5
PATH = sys.argv[1] if len(sys.argv) > 1 else "data/resources/references.json"
NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
BINS = int(sys.argv[3]) if len(sys.argv) > 3 else 48

# ---------- load ----------
raw = open(PATH, "rb").read()
keep = bytearray(b" " * 256)
for ch in b"0123456789.-":
    keep[ch] = ch
V = np.fromstring(raw.translate(bytes(keep)), sep=" ", dtype=np.float32).reshape(-1, DIMS)
parts = raw.split(b'"label":"')
is_fraud = (np.frombuffer(bytes(p[0] for p in parts[1:]), dtype=np.uint8) == ord("f"))
N = V.shape[0]
print(f"loaded N={N:,}")

# quantize exactly like src/vector.zig quantizeValue
Q = np.rint(np.clip(V, -1.0, 1.0) * SCALE).astype(np.int32)  # i32 to avoid overflow in math
Qf = Q.astype(np.int64)

# ---------- stage 1: bucket key ----------
online = (Q[:, 9] > SCALE // 2).astype(np.int64)
cardp = (Q[:, 10] > SCALE // 2).astype(np.int64)
unk = (Q[:, 11] > SCALE // 2).astype(np.int64)
hist = (Q[:, 5] > -SCALE // 2).astype(np.int64)
key = (online << 3) | (cardp << 2) | (unk << 1) | hist

# ---------- stage 2: per-bucket 2-D grid ----------
def build_bucket(idx):
    sub = Qf[idx]  # (n,14) int64
    var = sub.var(axis=0)
    var[[9, 10, 11]] = -1  # never grid on the bucket-defining bits
    da, db = np.argsort(var)[::-1][:2]
    a = sub[:, da]; b = sub[:, db]
    ea = np.linspace(a.min(), a.max() + 1, BINS + 1)
    eb = np.linspace(b.min(), b.max() + 1, BINS + 1)
    ca = np.clip(np.searchsorted(ea, a, side="right") - 1, 0, BINS - 1)
    cb = np.clip(np.searchsorted(eb, b, side="right") - 1, 0, BINS - 1)
    cell = ca * BINS + cb
    o = np.argsort(cell, kind="stable")
    cell_sorted = cell[o]
    members = idx[o]
    uniq, starts = np.unique(cell_sorted, return_index=True)
    ends = np.append(starts[1:], len(o))
    # per non-empty cell: (cell_id, member_slice_start, end, lo_a, hi_a, lo_b, hi_b)
    cells = []
    for cid, s, e in zip(uniq, starts, ends):
        cia, cib = divmod(int(cid), BINS)
        cells.append((cia, cib, members[s:e]))
    return {"da": da, "db": db, "ea": ea, "eb": eb, "cells": cells}

buckets = {}
for k in np.unique(key):
    buckets[int(k)] = build_bucket(np.nonzero(key == k)[0])

def interval_gap(q, lo, hi):
    # squared distance from point q to interval [lo, hi)
    if q < lo:
        return (lo - q) ** 2
    if q >= hi:
        return (q - (hi - 1)) ** 2
    return 0

def grid_search(qi):
    """Returns (top5_indices, candidates_scanned) for query row qi."""
    q = Qf[qi]
    bk = buckets[int(key[qi])]
    da, db, ea, eb = bk["da"], bk["db"], bk["ea"], bk["eb"]
    qa, qb = int(q[da]), int(q[db])
    # rank cells by lower bound, then scan with branch-and-bound
    ranked = []
    for cia, cib, mem in bk["cells"]:
        lb = interval_gap(qa, ea[cia], ea[cia + 1]) + interval_gap(qb, eb[cib], eb[cib + 1])
        ranked.append((lb, mem))
    ranked.sort(key=lambda t: t[0])
    best_d = np.full(K, np.iinfo(np.int64).max, dtype=np.int64)
    best_i = np.full(K, -1, dtype=np.int64)
    scanned = 0
    for lb, mem in ranked:
        if lb >= best_d.max():
            break
        m = mem[mem != qi]  # exclude self
        if m.size == 0:
            continue
        diff = Qf[m] - q
        d = np.einsum("ij,ij->i", diff, diff)
        scanned += m.size
        # merge into top-K
        cat_d = np.concatenate([best_d, d])
        cat_i = np.concatenate([best_i, m])
        sel = np.argpartition(cat_d, K)[:K]
        best_d, best_i = cat_d[sel], cat_i[sel]
    order = np.argsort(best_d)
    return best_i[order], scanned

def brute_topk(qi, data):
    q = data[qi]
    diff = data - q
    d = np.einsum("ij,ij->i", diff, diff)
    d[qi] = np.iinfo(d.dtype).max if np.issubdtype(d.dtype, np.integer) else np.inf
    nn = np.argpartition(d, K)[:K]
    return nn[np.argsort(d[nn])]

def decision(indices):
    f = int(is_fraud[indices].sum())
    return f / K < 0.6  # True = approved

# ---------- run ----------
rng = np.random.default_rng(7)
qidx = rng.choice(N, size=NQ, replace=False)
Vf = V.astype(np.float32)

t0 = time.time()
grid_mismatch_vs_i16 = 0
dec_mismatch_grid_i16 = 0
dec_mismatch_grid_float = 0
dec_mismatch_i16_float = 0
cand_stage1 = np.empty(NQ, dtype=np.int64)   # candidates if we scanned the whole bucket
cand_grid = np.empty(NQ, dtype=np.int64)     # candidates the grid actually scans
for n, qi in enumerate(qidx):
    g_idx, scanned = grid_search(qi)
    i16_idx = brute_topk(qi, Qf)
    f_idx = brute_topk(qi, Vf)
    cand_grid[n] = scanned
    cand_stage1[n] = (key == key[qi]).sum() - 1
    if set(g_idx.tolist()) != set(i16_idx.tolist()):
        grid_mismatch_vs_i16 += 1
    dg, di, df = decision(g_idx), decision(i16_idx), decision(f_idx)
    dec_mismatch_grid_i16 += (dg != di)
    dec_mismatch_grid_float += (dg != df)
    dec_mismatch_i16_float += (di != df)
dt = time.time() - t0

def pct(x):
    return f"{100*x/NQ:.3f}%"

print(f"\nran {NQ} held-out queries in {dt:.1f}s  (BINS={BINS})")
print("\n[1] EXACTNESS (grid+BBD vs i16 brute force):")
print(f"    top-5 set mismatches : {grid_mismatch_vs_i16}/{NQ} ({pct(grid_mismatch_vs_i16)})")
print(f"    decision  mismatches : {dec_mismatch_grid_i16}/{NQ} ({pct(dec_mismatch_grid_i16)})  <- MUST be 0")
print("\n[2] QUANTIZATION COST (i16-exact decision vs float-exact decision):")
print(f"    decision mismatches  : {dec_mismatch_i16_float}/{NQ} ({pct(dec_mismatch_i16_float)})")
print("\n[3] TOTAL vs float ground truth (grid decision vs float-exact decision):")
print(f"    decision mismatches  : {dec_mismatch_grid_float}/{NQ} ({pct(dec_mismatch_grid_float)})")
print("\n[4] CANDIDATES SCANNED PER QUERY:")
print(f"    stage-1 only (bucket): p50={np.percentile(cand_stage1,50):,.0f} "
      f"p99={np.percentile(cand_stage1,99):,.0f} max={cand_stage1.max():,}")
print(f"    stage-1 + grid       : p50={np.percentile(cand_grid,50):,.0f} "
      f"p99={np.percentile(cand_grid,99):,.0f} max={cand_grid.max():,}")
print(f"    grid speedup at p99  : {np.percentile(cand_stage1,99)/max(np.percentile(cand_grid,99),1):.1f}x"
      f"   vs full brute force: {N/max(np.percentile(cand_grid,99),1):,.0f}x")
