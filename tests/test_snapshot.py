"""Snapshot / restore tests — time-travel via crystal commutativity."""
from __future__ import annotations

from crystal_mem import CrystalMem


def test_snapshot_then_restore():
    m = CrystalMem(dim=512, n_heads=2)
    m.add("entry A")
    m.add("entry B")
    snap = m.snapshot()

    m.add("entry C")
    m.add("entry D")
    assert len(m.entries) == 4

    m.restore(snap)
    assert len(m.entries) == 2
    contents = {e.content for e in m.get_all()}
    assert contents == {"entry A", "entry B"}


def test_snapshot_preserves_entities():
    m = CrystalMem(dim=512, n_heads=2)
    m.add("alice fact 1", entity="alice")
    m.add("alice fact 2", entity="alice")
    snap = m.snapshot()

    m.forget_entity("alice")
    assert len(m.entries) == 0

    m.restore(snap)
    assert len(m.entity_facts("alice")) == 2
