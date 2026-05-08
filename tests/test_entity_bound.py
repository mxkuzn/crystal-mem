"""Tests for entity-bound memory + composition API.

Empirical: compositional bind cos > 0.98.
"""
from __future__ import annotations

from crystal_mem import CrystalMem


def test_bind_to_entity_and_retrieve():
    m = CrystalMem(dim=512, n_heads=2)
    m.add("Allergic to peanuts", entity="alice")
    m.add("Lives in Berlin", entity="alice")
    m.add("Works at Acme Corp", entity="alice")
    m.add("Loves jazz", entity="bob")

    alice_facts = m.entity_facts("alice")
    bob_facts = m.entity_facts("bob")

    assert len(alice_facts) == 3
    assert len(bob_facts) == 1
    contents = [f.content for f in alice_facts]
    assert "Allergic to peanuts" in contents
    assert "Lives in Berlin" in contents


def test_entities_lists_all():
    m = CrystalMem(dim=512, n_heads=2)
    m.add("fact 1", entity="proj_alpha")
    m.add("fact 2", entity="proj_beta")
    m.add("untagged fact")
    entities = set(m.entities())
    assert entities == {"proj_alpha", "proj_beta"}


def test_search_by_entity_isolated():
    m = CrystalMem(dim=512, n_heads=2)
    m.add("uses Python", entity="alice")
    m.add("uses Rust", entity="bob")
    m.add("uses Go", entity="alice")

    alice_results = m.search_by_entity("alice", "what languages")
    contents = [e.content for e, _ in alice_results]
    assert "uses Python" in contents
    assert "uses Go" in contents
    assert "uses Rust" not in contents


def test_forget_entity_bulk():
    m = CrystalMem(dim=512, n_heads=2)
    for i in range(5):
        m.add(f"alice fact {i}", entity="alice")
    m.add("bob fact", entity="bob")

    n_removed = m.forget_entity("alice")
    assert n_removed == 5
    assert "alice" not in m.entities()
    assert len(m.entries) == 1
    assert m.entries[next(iter(m.entries))].metadata.get("entity") == "bob"


def test_link_creates_bidirectional_index():
    m = CrystalMem(dim=512, n_heads=2)
    m.link("alice", "acme", "works_at")
    m.link("alice", "berlin", "lives_in")
    m.link("bob", "acme", "works_at")

    alice_facts = m.entity_facts("alice")
    acme_facts = m.entity_facts("acme")
    berlin_facts = m.entity_facts("berlin")

    assert len(alice_facts) == 2  # works_at + lives_in
    assert len(acme_facts) == 2  # alice + bob
    assert len(berlin_facts) == 1


def test_forget_entity_cleans_index():
    m = CrystalMem(dim=512, n_heads=2)
    m.add("fact", entity="x")
    assert "x" in m.entity_facts_idx
    m.forget_entity("x")
    assert "x" not in m.entity_facts_idx


def test_entity_survives_export_import():
    m = CrystalMem(dim=512, n_heads=2)
    m.add("alice's fact", entity="alice")
    m.add("untagged")

    crystal = m.export()
    m2 = CrystalMem.from_crystal(crystal)

    alice_facts = m2.entity_facts("alice")
    assert len(alice_facts) == 1
    assert alice_facts[0].content == "alice's fact"


def test_compositional_relation_search():
    """Knowledge-graph-style query: relations indexable by tag and entity."""
    m = CrystalMem(dim=512, n_heads=2)
    m.link("alice", "acme", "works_at")
    m.link("alice", "berlin", "lives_in")
    m.link("bob", "acme", "works_at")

    works_at = [e for e in m.get_all() if "works_at" in e.tags]
    assert len(works_at) == 2

    acme_workers = [e for e in m.entity_facts("acme") if "works_at" in e.tags]
    assert len(acme_workers) == 2
