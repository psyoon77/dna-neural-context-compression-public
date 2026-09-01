#!/usr/bin/env python3
import sys

K = 32

if len(sys.argv) == 2:
    K = int(sys.argv[1])

# Takes about 8 minutes for HoSa
output = 0
skipped = 0
kmer = sys.stdin.read(K)
while True:
    c = sys.stdin.read(1)
    # print(c, len(c), type(c))
    if len(c) == 0:
        break
    if c not in ["A", "C", "G", "T"]:
        skipped += 1
        continue
    kmer = kmer[1:] + c
    # print(f"{kmer=}")
    onehot = kmer
    onehot = onehot.replace("A", "0,0,0,1,")
    onehot = onehot.replace("C", "0,0,1,0,")
    onehot = onehot.replace("G", "0,1,0,0,")
    onehot = onehot.replace("T", "1,0,0,0,")
    onehot = onehot.rstrip(",")
    print(onehot)
    output += 1

print(f"Output {output} records, skipped {skipped} characters", file=sys.stderr)
