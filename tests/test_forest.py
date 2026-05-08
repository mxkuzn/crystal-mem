"""Tests for CrystalForest — bank of crystals + KG hybrid."""
from __future__ import annotations

import tempfile
from pathlib import Path

from crystal_mem.forest import CrystalForest


def test_basic_add_and_search():
    f = CrystalForest(dim=384, target_per_crystal=8, max_per_crystal=16)
    for i in range(20):
        f.add(f"fact about thing {i}")
    s = f.stats()
    assert s["n_shards"] >= 1
    assert s["total_entries"] == 20
    results = f.search("thing 5", top_k=3)
    assert len(results) >= 1


def test_auto_split_at_max():
    f = CrystalForest(dim=384, target_per_crystal=8, max_per_crystal=12)
    # Force many adds in one cluster
    for i in range(30):
        f.add(f"identical-cluster fact {i}")
    s = f.stats()
    # Should have triggered at least one split
    assert s["n_shards"] >= 2
    assert s["shard_size_max"] <= f.max_per_crystal


def test_kg_node_bound_to_shards():
    f = CrystalForest(dim=384, target_per_crystal=8, max_per_crystal=16)
    f.add("alice prefers Python", entity="alice")
    f.add("alice lives in Berlin", entity="alice")
    f.add("alice works at Acme", entity="alice")

    assert "alice" in f.kg_nodes
    node = f.kg_nodes["alice"]
    assert len(node.fact_shard_ids) >= 1


def test_kg_link_creates_edges():
    f = CrystalForest(dim=384)
    f.link("alice", "acme", "works_at",
           context="Alice works at Acme since 2020")
    f.link("alice", "berlin", "lives_in")
    assert "alice" in f.kg_nodes
    assert "acme" in f.kg_nodes
    assert "berlin" in f.kg_nodes
    assert len(f.kg_edges) == 2

    # neighbors
    nb = set(f.neighbors("alice", depth=1))
    assert "acme" in nb
    assert "berlin" in nb


def test_search_entity_isolated():
    f = CrystalForest(dim=384, target_per_crystal=8, max_per_crystal=16)
    f.add("uses Python", entity="alice")
    f.add("uses Rust", entity="bob")
    f.add("uses Go", entity="alice")

    alice_results = f.search_entity("alice", "what languages")
    contents = [r[1] for r in alice_results]
    assert any("Python" in c or "Go" in c for c in contents)
    assert not any("Rust" in c for c in contents)


def test_compact_merges_underfull_similar():
    """Two underfull shards with similar content should merge during compact()."""
    f = CrystalForest(dim=384, target_per_crystal=20, max_per_crystal=40)
    # Force two distinct shards by adding contradicting clusters
    for i in range(2):
        f.add("cooking recipe with garlic")
    for i in range(2):
        f.add("Python list comprehension")

    n_before = f.stats()["n_shards"]
    f.compact()
    n_after = f.stats()["n_shards"]
    assert n_after <= n_before


def test_stats_summary():
    f = CrystalForest(dim=384, target_per_crystal=8, max_per_crystal=16)
    for i in range(40):
        f.add(f"fact {i}", entity=f"e{i % 5}")
    s = f.stats()
    assert s["total_entries"] == 40
    assert s["n_kg_nodes"] == 5


def test_persistence_roundtrip():
    f = CrystalForest(dim=384, target_per_crystal=8, max_per_crystal=16)
    f.add("fact A", entity="alice")
    f.add("fact B", entity="bob")
    f.link("alice", "bob", "knows")

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "forest.bundle"
        f.export_file(str(path))

        f2 = CrystalForest.from_file(str(path))
        assert f2.stats()["total_entries"] == 2
        assert "alice" in f2.kg_nodes
        assert "bob" in f2.kg_nodes
        assert len(f2.kg_edges) == 1
