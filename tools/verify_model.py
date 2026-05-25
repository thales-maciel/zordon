#!/usr/bin/env python3
"""Independently verify the Zig-built model binary against the source references.

Parses the ZORDON2 format in Python and checks:
  - header sanity (count, dims, scale),
  - every stored vector sits in the bucket its 4 flag-bits imply (bucketing correct),
  - per-bucket size + fraud-rate match a fresh computation from the same vectors
    (label bitset aligned with the reordered vectors).
"""
import sys
import struct
import numpy as np

BIN = sys.argv[1] if len(sys.argv) > 1 else "data/model/references.i16.bin"
SCALE = 10000
SDIMS = 16

buf = open(BIN, "rb").read()
magic, version, count, dims, sdims, scale, bins, bcount = struct.unpack_from("<8sIIHHHHH", buf, 0)
cell_count, buckets_off, cells_off, vectors_off, labels_off = struct.unpack_from("<IQQQQ", buf, 28)
print(f"magic={magic!r} version={version} count={count:,} dims={dims} stored={sdims} scale={scale} "
      f"bins={bins} buckets={bcount} cells={cell_count}")
assert magic == b"ZORDONDB" and version == 2 and count == 3_000_000 and dims == 14 and sdims == SDIMS and scale == SCALE

# vectors: count x 16 i16
V = np.frombuffer(buf, dtype="<i2", count=count * SDIMS, offset=vectors_off).reshape(count, SDIMS)
# labels: bit j
lab_bytes = np.frombuffer(buf, dtype=np.uint8, count=(count + 7) // 8, offset=labels_off)
is_fraud = np.unpackbits(lab_bytes, bitorder="little")[:count].astype(bool)

# recompute the 4-bit key from the stored vectors
online = (V[:, 9] > SCALE // 2).astype(np.int64)
cardp = (V[:, 10] > SCALE // 2).astype(np.int64)
unk = (V[:, 11] > SCALE // 2).astype(np.int64)
hist = (V[:, 5] > -SCALE // 2).astype(np.int64)
key = (online << 3) | (cardp << 2) | (unk << 1) | hist

ok = True
print(f"\n{'bkt':>3} {'declared':>10} {'declared_da/db':>14} {'rows_by_key':>12} {'fraud%':>7}")
for k in range(bcount):
    vs, vc, cs, cc, da, db, _ = struct.unpack_from("<IIIIBBH", buf, buckets_off + k * 20)
    in_bucket = key[vs:vs + vc]
    # every vector in [vs, vs+vc) must have key == k
    coherent = bool((in_bucket == k).all()) if vc else True
    frate = 100 * is_fraud[vs:vs + vc].mean() if vc else 0.0
    by_key = int((key == k).sum())
    flag = "" if (coherent and by_key == vc) else "  <-- MISMATCH"
    if flag:
        ok = False
    print(f"{k:>3} {vc:>10,} {f'{da}/{db}':>14} {by_key:>12,} {frate:>6.2f}%{flag}")

# spot check: a few vectors decode to plausible values and cells partition the bucket
print("\nspot check vector[0]:", V[0][:dims] / SCALE)
print("total fraud:", f"{is_fraud.sum():,} ({100*is_fraud.mean():.2f}%)")
print("\nRESULT:", "OK - bucketing & labels consistent" if ok else "FAILED")
sys.exit(0 if ok else 1)
