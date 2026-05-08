"""04 — entity-bound memory + knowledge-graph links.

Bind facts to a named entity (person, project, topic). Query within entity
scope. Link entities with relations to build a small KG.

Empirical compositional bind cos > 0.98 (see `tests/test_entity_bound.py`).
"""
from crystal_mem import CrystalMem


m = CrystalMem(dim=512, n_heads=4)

# Per-entity facts
m.add("uses Python for backend",     entity="alice", tags={"work"})
m.add("lives in Berlin",             entity="alice", tags={"personal"})
m.add("uses Rust for systems work",  entity="bob",   tags={"work"})
m.add("collects vintage cameras",    entity="bob",   tags={"hobby"})

# All facts about a single entity
print("All facts about alice:")
for fact in m.entity_facts("alice"):
    print(f"  - {fact.content}")

# Entity-scoped search — won't leak across entities
print("\nWhat languages does alice use?")
for entry, score in m.search_by_entity("alice", "programming languages", top_k=3):
    print(f"  [{score:+.3f}] {entry.content}")

# Knowledge-graph links — entities with relations
m.link("alice", "acme",   relation="works_at")
m.link("alice", "berlin", relation="lives_in")
m.link("bob",   "acme",   relation="works_at")

# Acme has both alice and bob bound to it (bidirectional index)
print("\nWho's at acme?")
for fact in m.entity_facts("acme"):
    print(f"  - {fact.content}")

# Bulk forget: delete everything about alice (GDPR right-to-erasure)
removed = m.forget_entity("alice")
print(f"\nForgot {removed} facts about alice. Remaining: {len(m.entries)}")
