"""01 — quickstart: add, search, forget.

The 30-second tour of CrystalMem. No external models needed (uses fast
hash-based fallback embedder). Run:

    python examples/01_quickstart.py
"""
from crystal_mem import CrystalMem


# Create memory. dim=512 holds ~32 entries before HRR-bottleneck recall
# starts degrading; bank-mode (default search) has no such limit.
m = CrystalMem(user_id="alice", dim=512, n_heads=4)

# Write some facts
m.add("I prefer Python over Go for backend services")
m.add("My team's coding style: explicit > implicit")
m.add("Allergic to peanuts")
print(f"stored {len(m.entries)} memories")

# Find by semantic similarity
print("\nQuery: 'food restrictions'")
for entry, score in m.search("food restrictions", top_k=2):
    print(f"  [{score:+.3f}] {entry.content}")

# Find by id
some_id = list(m.entries.keys())[0]
print(f"\nGet by id: {m.get(some_id).content!r}")

# Forget (math-exact subtract — empirical Δ ≈ 4×10⁻⁶)
m.forget(some_id)
print(f"\nafter forget: {len(m.entries)} memories")
