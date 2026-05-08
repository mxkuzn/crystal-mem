"""Persistence (auto-save + auto-load) and thread-safety tests."""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from crystal_mem import CrystalMem


def test_persist_writes_on_add():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "state.crystal"
        m = CrystalMem(dim=512, n_heads=2, persist_path=str(path))
        m.add("first")
        assert path.exists()
        size1 = path.stat().st_size
        m.add("second")
        size2 = path.stat().st_size
        assert size2 >= size1


def test_persist_writes_on_forget():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "state.crystal"
        m = CrystalMem(dim=512, n_heads=2, persist_path=str(path))
        mid = m.add("temp")
        m.forget(mid)
        # File should still exist + be valid
        m2 = CrystalMem.from_file(path)
        assert len(m2.entries) == 0


def test_auto_load_restores_state():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "state.crystal"
        m1 = CrystalMem(dim=512, n_heads=2, persist_path=str(path))
        mid = m1.add("survived restart", entity="alice")

        # New instance with same path → auto-loads
        m2 = CrystalMem(dim=512, n_heads=2, persist_path=str(path))
        assert mid in m2.entries
        assert m2.entries[mid].content == "survived restart"
        assert m2.entity_facts("alice")


def test_atomic_write_no_partial_state():
    """If process is killed mid-write, the .tmp file is partial but main is intact."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "state.crystal"
        m = CrystalMem(dim=512, n_heads=2, persist_path=str(path))
        m.add("entry 1")
        # Simulate that an attempt left a leftover .tmp — main file should still be valid
        (path.with_suffix(".tmp")).write_bytes(b"corrupt")
        m.add("entry 2")  # subsequent write rename should overwrite
        m2 = CrystalMem.from_file(path)
        assert len(m2.entries) == 2


def test_concurrent_adds_no_data_loss():
    m = CrystalMem(dim=512, n_heads=2)
    n_threads = 8
    n_per_thread = 25

    def worker(tid: int):
        for i in range(n_per_thread):
            m.add(f"thread {tid} fact {i}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(m.entries) == n_threads * n_per_thread


def test_concurrent_add_forget_consistent():
    """No deadlock and entity index stays consistent."""
    m = CrystalMem(dim=512, n_heads=2)
    barrier = threading.Barrier(2)
    errors = []

    def adder():
        barrier.wait()
        for i in range(50):
            try:
                m.add(f"add {i}", entity="shared")
            except Exception as e:
                errors.append(e)

    def forgetter():
        barrier.wait()
        for _ in range(50):
            try:
                ids = list(m.entries.keys())
                if ids:
                    m.forget(ids[0])
            except Exception as e:
                errors.append(e)

    t1 = threading.Thread(target=adder)
    t2 = threading.Thread(target=forgetter)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"Concurrent ops raised: {errors}"
    # entity index is consistent: every id in the index actually exists
    for ent, ids in m.entity_facts_idx.items():
        for mid in ids:
            assert mid in m.entries
