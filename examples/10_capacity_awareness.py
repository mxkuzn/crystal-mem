"""10 — capacity awareness: honor the empirical recall law.

Each block-octonion crystal head holds about cap50 = dim/16 entries before
HRR-bottleneck recall starts degrading. Multi-head ensemble lifts this:
recall ≈ 1 − exp(−0.139·h·dim/N) with R² ≈ 0.97 (see `bench/capacity.py`).

For large N use one of:
    1. Bigger dim (linear cost in memory)
    2. More heads (linear cost in compute)
    3. CrystalForest — auto-sharded bank of crystals

Note: the standard `m.search()` uses the embedding bank (cosine over stored
embeddings), which is unaffected by cap50. The capacity warning matters for
HRR-bottleneck retrieval (`m.recall_via_crystal()`), privacy-mode storage,
and forwards-projecting capacity utilization to users.
"""
from crystal_mem import CrystalMem


# Small dim to demonstrate the bound clearly
m = CrystalMem(dim=128, n_heads=4)
print(f"cap50 per head = {m.cap50}")  # 128 / 16 = 8

# Add facts
for i in range(20):
    m.add(f"fact number {i}")

s = m.stats()
print("\nstats:")
for k, v in s.items():
    print(f"  {k:20s} {v}")

# Capacity warning kicks in past 70% of cap50
warning = m.capacity_warning()
if warning:
    print(f"\n{warning}")

# Capacity-bound retrieval (HRR-bottleneck — without bank shortcut)
ids = list(m.entries.keys())
correct = 0
for memory_id in ids:
    entry, _ = m.recall_via_crystal(memory_id)
    if entry and entry.id == memory_id:
        correct += 1
print(f"\nHRR-bottleneck recall: {correct}/{len(ids)} = {correct/len(ids):.0%}")
print(f"Recall-law prediction: {m.recall_estimate():.0%}")

# Standard bank-mode search is fine — full retrieval, no degradation
results = m.search("fact 10", top_k=3)
print("\nBank-mode search top-3 for 'fact 10':")
for entry, score in results:
    print(f"  [{score:+.3f}] {entry.content}")
