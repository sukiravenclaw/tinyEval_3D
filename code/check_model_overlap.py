"""
check_model_overlap.py
Computes pairwise prompt overlap between models using the binary item matrix
(item_matrix.csv is already a 0/1 pass/fail matrix, dichotomized upstream by
prep_r_data.py).
"""
import pandas as pd
import numpy as np
from itertools import combinations

mat = pd.read_csv("results/r_data/item_matrix.csv", index_col=0)
binary = mat.astype(int)
models = binary.index.tolist()
N = mat.shape[1]

print(f"Item matrix: {len(models)} models × {N} prompts (binary pass/fail)\n")

# ── Per-model pass counts ─────────────────────────────────────────────────────
print("=== Per-model pass rate ===")
for m in models:
    n_pass = binary.loc[m].sum()
    print(f"  {m:12s}: {n_pass:3d}/{N} pass  ({n_pass/N*100:.1f}%)")

# ── Pairwise overlap ──────────────────────────────────────────────────────────
print("\n=== Pairwise overlap (prompts where BOTH models pass) ===")
print(f"{'':14s}", end="")
for m in models:
    print(f"{m:>12s}", end="")
print()
for m1 in models:
    print(f"  {m1:12s}", end="")
    for m2 in models:
        if m1 == m2:
            n = binary.loc[m1].sum()
        else:
            n = (binary.loc[m1] & binary.loc[m2]).sum()
        print(f"{n:12d}", end="")
    print()

# ── Jaccard similarity matrix ─────────────────────────────────────────────────
print("\n=== Jaccard similarity (intersection / union) ===")
print(f"{'':14s}", end="")
for m in models:
    print(f"{m:>12s}", end="")
print()
for m1 in models:
    print(f"  {m1:12s}", end="")
    for m2 in models:
        if m1 == m2:
            print(f"{'1.000':>12s}", end="")
        else:
            inter = (binary.loc[m1] & binary.loc[m2]).sum()
            union = (binary.loc[m1] | binary.loc[m2]).sum()
            j = inter / union if union > 0 else 0.0
            print(f"{j:12.3f}", end="")
    print()

# ── Unique prompts per model (only that model passes) ─────────────────────────
print("\n=== Prompts passed by exactly ONE model (unique items) ===")
for m in models:
    others = [x for x in models if x != m]
    other_pass = binary.loc[others].sum(axis=0)
    unique = binary.loc[m] & (other_pass == 0)
    print(f"  {m:12s}: {unique.sum():3d} unique prompts  "
          f"(prompt IDs: {list(binary.columns[unique.values.astype(bool)])[:10]}"
          f"{'...' if unique.sum() > 10 else ''})")

# ── Prompts no model passes ───────────────────────────────────────────────────
none_pass = (binary.sum(axis=0) == 0)
all_pass  = (binary.sum(axis=0) == len(models))
print(f"\n  Prompts where NO model passes:  {none_pass.sum()}")
print(f"  Prompts where ALL models pass:  {all_pass.sum()}")

# ── Zero-overlap pairs ────────────────────────────────────────────────────────
print("\n=== Pairs with zero overlapping pass items ===")
found = False
for m1, m2 in combinations(models, 2):
    inter = (binary.loc[m1] & binary.loc[m2]).sum()
    if inter == 0:
        print(f"  {m1} ∩ {m2} = 0 prompts")
        found = True
if not found:
    print("  None — all pairs share at least one passing prompt.")
