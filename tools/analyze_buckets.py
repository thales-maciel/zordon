#!/usr/bin/env python3
"""Within-bucket spread analysis to choose the second-level partition dimension.

Loads vectors the same way as analyze_refs.py (no NN sampling -> fast).
For the largest 4-bit buckets, reports per-dimension std so we can pick which
continuous dim best subdivides the big buckets for an exact sort/grid prune.
"""
import sys
import numpy as np

DIMS = 14
PATH = sys.argv[1] if len(sys.argv) > 1 else "data/resources/references.json"
NAMES = ["amount", "installments", "amount_vs_avg", "hour_of_day", "day_of_week",
         "min_since_last", "km_from_last", "km_from_home", "tx_count_24h",
         "is_online", "card_present", "unknown", "mcc_risk", "merchant_avg"]

raw = open(PATH, "rb").read()
keep = bytearray(b" " * 256)
for ch in b"0123456789.-":
    keep[ch] = ch
V = np.fromstring(raw.translate(bytes(keep)), sep=" ", dtype=np.float32).reshape(-1, DIMS)
N = V.shape[0]

online = (V[:, 9] > 0.5).astype(np.int64)
cardp = (V[:, 10] > 0.5).astype(np.int64)
unk = (V[:, 11] > 0.5).astype(np.int64)
hist = (V[:, 5] > -0.5).astype(np.int64)
key = (online << 3) | (cardp << 2) | (unk << 1) | hist
counts = np.bincount(key, minlength=16)
order = [k for k in np.argsort(counts)[::-1] if counts[k] > 0]

# continuous dims worth sorting on (exclude the 4 bucket bits and sentinel dim5)
cont = [0, 2, 3, 4, 6, 7, 8, 12, 13]
# Effective d_5 window to prune against (use observed p99 from analyze_refs.py):
W = np.sqrt(0.043)  # ~0.207

print(f"second-level partition analysis (prune half-window w=sqrt(d5^2 p99)={W:.3f})")
print("for each big bucket, per-dim std and est. fraction kept if we sort by that dim")
print("(fraction kept ~ avg count within +/- w of a query point, lower is better)\n")

for k in order[:4]:
    mask = key == k
    B = V[mask]
    n = B.shape[0]
    print(f"bucket {k}: {n:,} rows ({100*n/N:.2f}%)")
    rows = []
    for d in cont:
        col = np.sort(B[:, d])
        std = col.std()
        # exact expected kept-fraction for a sorted 1-D prune of half-width W:
        lo = np.searchsorted(col, col - W, side="left")
        hi = np.searchsorted(col, col + W, side="right")
        kept = (hi - lo).mean() / n
        rows.append((kept, d, std))
    rows.sort()
    for kept, d, std in rows:
        print(f"    dim {d:2d} {NAMES[d]:<15} std={std:.4f}  sort-prune keeps ~{100*kept:5.2f}%")
    print()
