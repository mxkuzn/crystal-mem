"""06 — federated merge: combine multiple memories cleanly.

HRR sum is commutative + associative, so merging crystals is mathematically
equivalent to building a single crystal from all the source entries.
Empirical Δ vs centralized rebuild: ~8.6×10⁻⁸ (see `demos/d3_team_shared_pool.py`).

Use cases:
    - Team brain (each dev's memory merged at end of week)
    - Multi-source RAG with provenance
    - Federated learning where raw data shouldn't leave its source
"""
from crystal_mem import CrystalMem


# Three independent memories
alice = CrystalMem(dim=512, n_heads=2, user_id="alice")
alice.add("Found 10× more null users than expected in prod",
          entity="prod_data", tags={"finding"})

bob = CrystalMem(dim=512, n_heads=2, user_id="bob")
bob.add("Migration script needs idempotent rerun handling",
        entity="phoenix", tags={"todo"})

carol = CrystalMem(dim=512, n_heads=2, user_id="carol")
carol.add("Rate limiter trips at 500 req/min, not 1000 as documented",
          entity="prod_data", tags={"finding"})

# Combine into a team brain — no conflict resolution needed
team = CrystalMem(dim=512, n_heads=2, user_id="team")
team.merge(alice)
team.merge(bob)
team.merge(carol)

print(f"team brain: {len(team.entries)} entries from 3 sources")
for entry in team.get_all():
    print(f"  [{entry.user_id}] {entry.content}")

# Bob leaves the team — surgically remove his contributions
team.unmerge(bob)
print(f"\nafter bob's exit: {len(team.entries)} entries remain")

# Verify search still works on remaining shards
print("\nQuery 'production observations':")
for entry, score in team.search("production observations", top_k=3):
    print(f"  [{score:+.3f}] {entry.content}")
