"""09 — migrate an existing Mem0 dump into CrystalMem.

CrystalMem ships with adapters for Mem0, Letta, Zep, and chat-jsonl exports.
This example uses a synthetic Mem0-shape dump; in practice you'd download
your real export from those platforms.

CLI equivalent:
    python -m crystal_mem.migrate \\
        --source mem0 --input dump.json \\
        --output alice.crystal --dim 384
"""
from crystal_mem import CrystalMem
from crystal_mem.migrate import migrate_from_mem0


# Mem0 export shape (real Mem0 returns this from `client.get_all()`)
mem0_dump = {
    "memories": [
        {
            "id": "m1",
            "memory": "user prefers vegan cuisine",
            "user_id": "alice",
            "created_at": "2026-04-01T10:00:00Z",
            "metadata": {"category": "dietary"},
        },
        {
            "id": "m2",
            "memory": "user lives in Berlin since 2019",
            "user_id": "alice",
            "created_at": "2026-04-02T08:30:00Z",
            "metadata": {"category": "location"},
        },
        {
            "id": "m3",
            "memory": "user works at Acme Corp as a backend engineer",
            "user_id": "alice",
            "created_at": "2026-04-05T14:15:00Z",
        },
    ]
}

# Ingest into a fresh CrystalMem
m = CrystalMem(user_id="alice", dim=512, n_heads=2)
n_imported = migrate_from_mem0(m, mem0_dump)
print(f"imported {n_imported} memories from Mem0")

# Imported entries carry their original metadata
for e in m.get_all():
    print(f"  - {e.content}")
    print(f"      mem0_id={e.metadata.get('mem0_id')}, "
          f"category={e.metadata.get('category', '—')}, "
          f"tags={sorted(e.tags)}")

# Now you can use CrystalMem features that Mem0 doesn't expose:
# • Exact forget — m.forget(memory_id)
# • Federated merge — m.merge(other)
# • Portable export — m.export_file("alice.crystal")
# • Watermark — watermark_inject(m, label=...)
print(f"\ntotal: {len(m.entries)} entries, ready for downstream use")
