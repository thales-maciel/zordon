#!/usr/bin/env python3
"""Analyze the contest reference dataset to inform index design.

The references file is a single JSON array of {"vector":[14 floats],"label":"fraud|legit"}.
Keys and labels contain no digits, so every numeric token in the file is a vector
component -> we strip all non-numeric bytes and reshape to (N, 14). Labels are read
separately by splitting on the label key.
"""
import sys
import time
import numpy as np

DIMS = 14
PATH = sys.argv[1] if len(sys.argv) > 1 else "data/resources/references.json"

# Dimension names per DETECTION_RULES.md
NAMES = [
    "amount", "installments", "amount_vs_avg", "hour_of_day", "day_of_week",
    "minutes_since_last", "km_from_last", "km_from_home", "tx_count_24h",
    "is_online", "card_present", "unknown_merchant", "mcc_risk", "merchant_avg",
]

t0 = time.time()
with open(PATH, "rb") as f:
    raw = f.read()
print(f"read {len(raw)/1e6:.0f} MB in {time.time()-t0:.1f}s")

# --- labels (ordered, aligned with records) ---
t0 = time.time()
parts = raw.split(b'"label":"')
# parts[0] is the prefix before the first label; each later part starts with f|l
labels = np.frombuffer(
    bytes(p[0] for p in parts[1:]), dtype=np.uint8
)
is_fraud = (labels == ord("f")).astype(np.uint8)
print(f"parsed {len(is_fraud)} labels in {time.time()-t0:.1f}s")

# --- vectors: keep only [0-9 . -], everything else -> space ---
t0 = time.time()
keep = bytearray(b" " * 256)
for ch in b"0123456789.-":
    keep[ch] = ch
cleaned = raw.translate(bytes(keep))
vals = np.fromstring(cleaned, sep=" ", dtype=np.float32)
print(f"parsed {vals.size} numbers in {time.time()-t0:.1f}s")
assert vals.size % DIMS == 0, f"{vals.size} not divisible by {DIMS}"
V = vals.reshape(-1, DIMS)
N = V.shape[0]
assert N == len(is_fraud), f"vectors {N} != labels {len(is_fraud)}"
print(f"\nN = {N:,} vectors, {DIMS} dims")

# === A. label balance ===
nf = int(is_fraud.sum())
print(f"\n[A] labels: fraud={nf:,} ({100*nf/N:.2f}%)  legit={N-nf:,} ({100*(N-nf)/N:.2f}%)")

# === C. per-dim cardinality / ranges ===
print("\n[C] per-dimension stats:")
print(f"{'idx':>3} {'name':<19} {'card':>8} {'min':>9} {'max':>9} {'p50':>9} {'p99':>9}")
card = []
for d in range(DIMS):
    col = V[:, d]
    u = np.unique(col)
    card.append(u.size)
    print(f"{d:>3} {NAMES[d]:<19} {u.size:>8} {col.min():>9.4f} {col.max():>9.4f} "
          f"{np.percentile(col,50):>9.4f} {np.percentile(col,99):>9.4f}")

# === B. bucket sizes from the 4 cheap discrete bits ===
# bits: is_online(9), card_present(10), unknown_merchant(11), has_history(dim5 != -1)
online = (V[:, 9] > 0.5).astype(np.int64)
cardp = (V[:, 10] > 0.5).astype(np.int64)
unk = (V[:, 11] > 0.5).astype(np.int64)
hist = (V[:, 5] > -0.5).astype(np.int64)  # 1 = has history, 0 = sentinel -1
key = (online << 3) | (cardp << 2) | (unk << 1) | hist
print("\n[B] 16-bucket sizes (online,card_present,unknown,has_history):")
counts = np.bincount(key, minlength=16)
order = np.argsort(counts)[::-1]
for k in order:
    if counts[k] == 0:
        continue
    o, c, u2, h = (k >> 3) & 1, (k >> 2) & 1, (k >> 1) & 1, k & 1
    frauds_in = int(is_fraud[key == k].sum())
    print(f"  bucket {k:2d} on={o} cp={c} unk={u2} hist={h}: "
          f"{counts[k]:>10,} ({100*counts[k]/N:5.2f}%)  fraud={100*frauds_in/max(counts[k],1):5.2f}%")
print(f"  biggest bucket = {counts.max():,} ({100*counts.max()/N:.2f}% of data)")

# secondary split candidate: combine 4 bits with hour_of_day (dim3) cardinality
print("\n[B2] biggest bucket sub-split by hour_of_day (dim3):")
big = order[0]
mask = key == big
sub = V[mask, 3]
subu, subc = np.unique(sub, return_counts=True)
print(f"  bucket {big}: {mask.sum():,} rows, {subu.size} distinct hours, "
      f"max sub-bucket = {subc.max():,} ({100*subc.max()/N:.2f}% of total data)")

# === D. 5-NN d_k^2 distribution + in-bucket validation ===
K = 5
SAMPLE = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
rng = np.random.default_rng(42)
qidx = rng.choice(N, size=SAMPLE, replace=False)
Vf = V.astype(np.float32)
t0 = time.time()
d5sq = np.empty(SAMPLE, dtype=np.float64)
same_bucket = np.zeros(SAMPLE, dtype=np.int64)  # how many of the 5 NN share the query bucket
prunable = 0  # d5^2 < 1.0  => one differing binary flag already excludes
for i, qi in enumerate(qidx):
    diff = Vf - Vf[qi]
    dsq = np.einsum("ij,ij->i", diff, diff)
    dsq[qi] = np.inf  # exclude self
    nn = np.argpartition(dsq, K)[:K]
    nn = nn[np.argsort(dsq[nn])]
    d5sq[i] = dsq[nn[-1]]
    same_bucket[i] = int((key[nn] == key[qi]).sum())
    if d5sq[i] < 1.0:
        prunable += 1
print(f"\n[D] {SAMPLE} sampled queries, exact 5-NN over {N:,} in {time.time()-t0:.1f}s")
print(f"  d_5^2 (squared dist of 5th NN): "
      f"min={d5sq.min():.4f} p50={np.percentile(d5sq,50):.4f} "
      f"p99={np.percentile(d5sq,99):.4f} max={d5sq.max():.4f}")
print(f"  queries with d_5^2 < 1.0 (one flag prunes everything else): "
      f"{prunable}/{SAMPLE} = {100*prunable/SAMPLE:.2f}%")
print(f"  of the 5 true NN, count sharing the query's 4-bit bucket: "
      f"mean={same_bucket.mean():.3f} / {K}  (all-5 in-bucket: "
      f"{100*(same_bucket==K).mean():.2f}%)")
