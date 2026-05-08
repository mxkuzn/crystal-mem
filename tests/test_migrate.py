"""Migration script tests — synthetic dumps in each format."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from crystal_mem import CrystalMem
from crystal_mem.migrate import (
    migrate_from_chat_jsonl,
    migrate_from_letta,
    migrate_from_mem0,
    migrate_from_zep,
)


def test_mem0_full_format():
    dump = {
        "memories": [
            {"id": "m1", "memory": "user prefers Python", "user_id": "alice",
             "created_at": "2026-04-01", "metadata": {"category": "lang"}},
            {"id": "m2", "memory": "user lives in Berlin", "user_id": "alice",
             "created_at": "2026-04-02", "metadata": {"category": "loc"}},
        ]
    }
    m = CrystalMem(dim=384, n_heads=2)
    n = migrate_from_mem0(m, dump)
    assert n == 2
    assert len(m.entries) == 2
    contents = [e.content for e in m.get_all()]
    assert "user prefers Python" in contents
    assert "user lives in Berlin" in contents


def test_mem0_flat_list_format():
    dump = [
        {"memory": "fact A"},
        {"content": "fact B"},
        {"text": "fact C"},
    ]
    m = CrystalMem(dim=384, n_heads=2)
    n = migrate_from_mem0(m, dump)
    assert n == 3


def test_mem0_skips_empty():
    dump = {"memories": [{"id": "x", "memory": ""}, {"id": "y"}]}
    m = CrystalMem(dim=384, n_heads=2)
    assert migrate_from_mem0(m, dump) == 0


def test_letta_format():
    dump = {
        "persona": "I am a helpful assistant",
        "human": "User is Alice, 32, software engineer",
        "archival_memory": [
            {"id": 1, "text": "First major project completed Q1"},
            {"id": 2, "text": "Promoted to senior in Q3"},
        ],
        "recall_memory": [
            {"text": "ephemeral chat snippet"},
        ],
    }
    m = CrystalMem(dim=384, n_heads=2)
    n = migrate_from_letta(m, dump, include_recall=False)
    assert n == 4  # persona + human + 2 archival
    n2 = migrate_from_letta(m, dump, include_recall=True)
    # persona + human are duplicate by content but different ids — should add
    assert n2 == 5  # persona + human + 2 archival + 1 recall


def test_zep_format():
    dump = {
        "messages": [
            {"role": "user", "content": "I like jazz"},
            {"role": "assistant", "content": "Noted!"},
            {"role": "user", "content": "Especially Brazilian jazz"},
        ]
    }
    m = CrystalMem(dim=384, n_heads=2)
    n = migrate_from_zep(m, dump)
    assert n == 3
    assert any("jazz" in e.content for e in m.get_all())


def test_chat_jsonl_file():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "chat.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"role": "user", "content": "hi there"}) + "\n")
            f.write("\n")  # empty line
            f.write(json.dumps({"role": "assistant", "content": "hello!"}) + "\n")

        m = CrystalMem(dim=384, n_heads=2)
        n = migrate_from_chat_jsonl(m, str(path))
        assert n == 2


def test_imported_facts_searchable():
    """After migration, entries are queryable like native ones."""
    dump = {"memories": [
        {"id": "a", "memory": "Python is my favorite"},
        {"id": "b", "memory": "I dislike Java"},
    ]}
    m = CrystalMem(dim=384, n_heads=2)
    migrate_from_mem0(m, dump)
    results = m.search("favorite programming language", top_k=2)
    assert len(results) >= 1
