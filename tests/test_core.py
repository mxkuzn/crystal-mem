"""Smoke tests for CrystalMem core API."""
from __future__ import annotations

import pytest

from crystal_mem import CrystalMem


def test_add_and_search():
    m = CrystalMem(user_id="alice", dim=512, n_heads=2)
    mid = m.add("user prefers Python")
    assert mid in m.entries
    results = m.search("user prefers Python", top_k=3)
    assert len(results) >= 1
    assert results[0][0].id == mid


def test_forget_removes_entry():
    m = CrystalMem(dim=512)
    mid = m.add("delete me")
    assert mid in m.entries
    m.forget(mid)
    assert mid not in m.entries


def test_forget_subtracts_from_crystal_exactly():
    """Math-grade forget — crystal returns to pre-add state."""
    m = CrystalMem(dim=512, n_heads=2)
    norms_before = [c.norm().item() for c in m.crystals]
    mid = m.add("temp")
    m.forget(mid)
    norms_after = [c.norm().item() for c in m.crystals]
    for nb, na in zip(norms_before, norms_after):
        assert abs(nb - na) < 1e-5


def test_capacity_warning():
    m = CrystalMem(dim=128)  # cap50 = 8
    for i in range(20):
        m.add(f"fact {i}")
    assert m.capacity_used > 1.0
    assert m.capacity_warning() is not None


def test_recall_law_in_range():
    m = CrystalMem(dim=1024, n_heads=4)
    for i in range(50):
        m.add(f"fact {i}")
    est = m.recall_estimate()
    assert 0.0 < est <= 1.0


def test_update_replaces_content():
    m = CrystalMem(dim=512)
    mid = m.add("old", tags={"x"}, metadata={"k": "v"})
    m.update(mid, "new")
    assert m.entries[mid].content == "new"
    assert "x" in m.entries[mid].tags
    assert m.entries[mid].metadata == {"k": "v"}


def test_mode_auto_requires_extractor():
    m = CrystalMem(mode="auto")
    with pytest.raises(RuntimeError):
        m.add_from_conversation([{"role": "user", "content": "hi"}])


def test_dim_must_be_multiple_of_8():
    with pytest.raises(AssertionError):
        CrystalMem(dim=100)
