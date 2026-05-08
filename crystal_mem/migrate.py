"""Migration utilities — import existing memory dumps from Mem0, Letta, Zep.

Each adapter takes a serialized export (JSON or dict) and ingests entries
into a CrystalMem instance, preserving timestamps and metadata where possible.

Quick start:
    from crystal_mem import CrystalMem
    from crystal_mem.migrate import migrate_from_mem0

    m = CrystalMem(user_id="alice", dim=384, embedder=...)
    n = migrate_from_mem0(m, "mem0_export.json")
    print(f"Imported {n} memories from Mem0")

CLI:
    python -m crystal_mem.migrate --source mem0 --input dump.json \\
                                  --output alice.crystal --dim 384
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


# -------------------------------------------------------------------------
# Mem0 — https://github.com/mem0ai/mem0
# Export shape: {"memories": [{"id", "memory", "user_id", "metadata",
#                              "created_at", "updated_at"}]}
# -------------------------------------------------------------------------
def migrate_from_mem0(memory, source: str | Path | dict) -> int:
    """Ingest entries from a Mem0-format export. Returns count imported.

    Mem0 export schema:
        {"memories": [{"id", "memory", "user_id", "metadata", "created_at", ...}, ...]}
    Or a flat list of memory dicts.
    """
    data = source if isinstance(source, (dict, list)) else _load_json(source)
    if isinstance(data, dict):
        items = data.get("memories", data.get("results", []))
    else:
        items = list(data)

    count = 0
    for item in items:
        content = item.get("memory") or item.get("content") or item.get("text")
        if not content:
            continue
        meta = dict(item.get("metadata") or {})
        meta["mem0_id"] = item.get("id")
        meta["mem0_created_at"] = item.get("created_at")
        meta["mem0_updated_at"] = item.get("updated_at")
        memory.add(
            content,
            tags={"imported", "mem0"},
            metadata=meta,
        )
        count += 1
    return count


# -------------------------------------------------------------------------
# Letta — https://github.com/letta-ai/letta
# Export shape: persona + human + archival_memory + recall_memory.
# We capture archival_memory entries (long-term) by default.
# -------------------------------------------------------------------------
def migrate_from_letta(memory, source: str | Path | dict,
                       include_recall: bool = False) -> int:
    """Ingest archival_memory + optionally recall_memory from a Letta export."""
    data = source if isinstance(source, dict) else _load_json(source)

    count = 0
    persona = data.get("persona")
    if persona:
        memory.add(persona, tags={"imported", "letta", "persona"},
                   metadata={"letta_field": "persona"})
        count += 1
    human = data.get("human")
    if human:
        memory.add(human, tags={"imported", "letta", "human"},
                   metadata={"letta_field": "human"})
        count += 1

    archival = data.get("archival_memory") or []
    if isinstance(archival, dict):
        archival = list(archival.values())
    for item in archival:
        content = item.get("text") or item.get("content") or str(item)
        memory.add(content, tags={"imported", "letta", "archival"},
                   metadata={"letta_id": item.get("id")} if isinstance(item, dict) else {})
        count += 1

    if include_recall:
        recall = data.get("recall_memory") or []
        for item in recall:
            content = item.get("text") or item.get("content") or str(item)
            memory.add(content, tags={"imported", "letta", "recall"})
            count += 1
    return count


# -------------------------------------------------------------------------
# Zep — https://github.com/getzep/zep
# Export shape: list of session messages or summary objects.
# -------------------------------------------------------------------------
def migrate_from_zep(memory, source: str | Path | dict | list) -> int:
    """Ingest entries from a Zep memory export."""
    data = source if isinstance(source, (dict, list)) else _load_json(source)
    if isinstance(data, dict):
        items = data.get("messages", data.get("summary", data.get("memory", [])))
    else:
        items = list(data)

    count = 0
    for item in items:
        if isinstance(item, str):
            memory.add(item, tags={"imported", "zep"})
            count += 1
            continue
        content = (
            item.get("content") or item.get("text")
            or item.get("summary") or item.get("message")
        )
        if not content:
            continue
        meta = {"zep_role": item.get("role"), "zep_id": item.get("uuid")}
        memory.add(content, tags={"imported", "zep"}, metadata=meta)
        count += 1
    return count


# -------------------------------------------------------------------------
# Generic — chat history JSONL (e.g. ChatGPT conversation exports)
# -------------------------------------------------------------------------
def migrate_from_chat_jsonl(memory, source: str | Path) -> int:
    """Each line: {"role": "user|assistant", "content": "..."}."""
    count = 0
    with open(source) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = item.get("content")
            if not content:
                continue
            memory.add(
                content,
                tags={"imported", "chat", item.get("role", "msg")},
                metadata={"role": item.get("role"),
                          "imported_from": str(source),
                          "imported_at": time.time()},
            )
            count += 1
    return count


def main():
    p = argparse.ArgumentParser(prog="crystal-mem-migrate")
    p.add_argument("--source", required=True, choices=["mem0", "letta", "zep", "chat-jsonl"])
    p.add_argument("--input", required=True, help="Path to input dump")
    p.add_argument("--output", required=True, help="Path to write output crystal")
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--user-id", default="migrated")
    p.add_argument("--no-embedder", action="store_true",
                   help="Use hash embedder (faster, no semantic match)")
    args = p.parse_args()

    from . import CrystalMem
    if args.no_embedder:
        from .embedders import hash_embedder
        emb = hash_embedder(target_dim=args.dim)
    else:
        from .embedders import sentence_transformer_embedder
        emb = sentence_transformer_embedder(target_dim=args.dim)

    m = CrystalMem(user_id=args.user_id, dim=args.dim, n_heads=args.heads,
                   embedder=emb)

    fn = {
        "mem0":       migrate_from_mem0,
        "letta":      migrate_from_letta,
        "zep":        migrate_from_zep,
        "chat-jsonl": migrate_from_chat_jsonl,
    }[args.source]

    n = fn(m, args.input)
    m.export_file(args.output)
    print(f"Migrated {n} entries from {args.source} → {args.output}")


if __name__ == "__main__":
    main()
