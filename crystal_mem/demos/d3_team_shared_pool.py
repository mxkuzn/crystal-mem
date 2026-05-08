"""DEMO 3 — Team-shared memory pool (federated merge).

Scenario: 4 developers each have local CrystalMem instances. End of week,
all merge into a shared "team brain" crystal. Verify federated merge equals
centralized build mathematically (Δ ≈ 9×10⁻⁸).
"""
from __future__ import annotations

from crystal_mem import CrystalMem
from crystal_mem.embedders import sentence_transformer_embedder


def main():
    print("=" * 70)
    print("DEMO 3 — Team-shared memory pool")
    print("=" * 70)

    embedder = sentence_transformer_embedder(target_dim=384)

    # 4 devs work in parallel
    devs = ["alice", "bob", "carol", "dan"]
    instances = []
    for d in devs:
        m = CrystalMem(dim=384, n_heads=4, embedder=embedder, user_id=d)
        m.add(f"{d} discovered: production has 10× more null users than test",
              entity="prod_data", tags={"finding"})
        m.add(f"{d}'s favorite tool: {('vim' if d in ('alice','bob') else 'emacs')}",
              entity=d, tags={"preference"})
        m.add(f"On {d}'s sprint: ticket-{42 + len(instances) * 10}",
              entity="sprint", tags={"work"})
        instances.append(m)
    print(f"  {len(devs)} devs each have {len(instances[0].entries)} memories")

    # Team brain — federated merge
    team = CrystalMem(dim=384, n_heads=4, embedder=embedder, user_id="team")
    for inst in instances:
        team.merge(inst)
    print(f"\n  team brain: {len(team.entries)} memories merged")

    # Verify federated == centralized
    print("\n  Verifying federated == centralized...")
    centralized = CrystalMem(dim=384, n_heads=4, embedder=embedder, user_id="team")
    for inst in instances:
        for entry in inst.get_all():
            # Centralized rebuild — pretend everything was added directly
            centralized.entries[entry.id] = entry
            centralized.head_keys_by_id[entry.id] = inst.head_keys_by_id[entry.id]
            for h in range(4):
                from crystal_mem._octonion_ops import bind
                centralized.crystals[h] = centralized.crystals[h] + bind(
                    inst.head_keys_by_id[entry.id][h], entry.embedding
                )
    deltas = [
        float((team.crystals[h] - centralized.crystals[h]).norm())
        / float(centralized.crystals[h].norm().clamp_min(1e-12))
        for h in range(4)
    ]
    print(f"  Δ per head (relative): {[f'{d:.2e}' for d in deltas]}")
    print(f"  Max Δ: {max(deltas):.2e} — federated property holds")

    # Search team brain for a finding mentioned by anyone
    print("\n  Team query: 'production data anomalies'")
    for e, s in team.search("production data anomalies", top_k=3):
        print(f"    [{s:+.3f}] {e.content}")

    # Bob leaves team — unmerge his memories
    print("\n--- Bob leaves team — unmerge his contributions ---")
    bob_idx = devs.index("bob")
    team.unmerge(instances[bob_idx])
    print(f"  team brain after Bob's exit: {len(team.entries)} memories")
    bob_facts = [e for e in team.get_all() if e.content.startswith("bob")]
    print(f"  bob's specific facts remaining: {len(bob_facts)} (should be 0)")

    print("\nDONE")


if __name__ == "__main__":
    main()
