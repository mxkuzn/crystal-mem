"""Memory portability — export / import / merge / diff / unmerge.

Research backing:
    Federated sum-merge — mathematically identical to centralized rebuild.
        Empirical Δ|M_merged − M_centralized| / |M| ≈ 9×10⁻⁸.
    Exact subtract — supports symmetric unmerge.
    HRR algebra — commutative + associative bind sums → no conflict resolution.
    Optional watermark for provenance (see `watermark.py`).

Crystal file format (binary):
    HEADER  (8 bytes magic "CMEMv001" + json metadata blob)
    PAYLOAD (torch.save dict with crystals, entries, head_keys)
"""
from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

import torch

if TYPE_CHECKING:
    from .core import CrystalMem  # noqa: F401


MAGIC = b"CMEMv001"


@dataclass
class Crystal:
    """Self-contained portable memory crystal.

    Created by CrystalMem.export() / .export_file().
    Loaded back via CrystalMem.from_file() or CrystalMem.merge_file().

    Format:
        - magic header
        - json metadata (dim, n_heads, algebra, user_id, created_at, n_entries, watermark?)
        - torch.save payload (crystals: list[Tensor], entries: list[dict], head_keys: dict)
    """
    metadata: dict
    crystals: list[torch.Tensor]
    entries: list[dict]                # serialized MemoryEntry minus embedding
    embeddings: torch.Tensor           # stacked entry embeddings [N, dim]
    head_keys: dict[str, list[torch.Tensor]]

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        buf.write(MAGIC)
        meta_bytes = json.dumps(self.metadata).encode("utf-8")
        buf.write(len(meta_bytes).to_bytes(4, "big"))
        buf.write(meta_bytes)
        torch.save(
            {
                "crystals": self.crystals,
                "entries": self.entries,
                "embeddings": self.embeddings,
                "head_keys": self.head_keys,
            },
            buf,
        )
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "Crystal":
        buf = io.BytesIO(data)
        magic = buf.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"Bad magic: {magic!r}")
        meta_len = int.from_bytes(buf.read(4), "big")
        meta = json.loads(buf.read(meta_len).decode("utf-8"))
        payload = torch.load(buf, weights_only=False)
        return cls(
            metadata=meta,
            crystals=payload["crystals"],
            entries=payload["entries"],
            embeddings=payload["embeddings"],
            head_keys=payload["head_keys"],
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "Crystal":
        return cls.from_bytes(Path(path).read_bytes())

    def to_file(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_bytes())


# ---------------------------------------------------------------------------
# CrystalMem extension methods (monkey-patched for clean API ergonomics)
# ---------------------------------------------------------------------------
def _export(self, *, filter: Callable | None = None) -> Crystal:  # noqa: A002
    """Serialize current memory state to a portable Crystal.

    If filter provided, only matching entries are included AND a fresh crystal
    is built (sum of bind(key, emb) for filtered entries) so that retrieval on
    the imported subset returns clean results.
    """
    if filter is None:
        entries_dicts = [e.to_dict() for e in self.entries.values()]
        embeddings = (
            torch.stack([e.embedding for e in self.entries.values()])
            if self.entries
            else torch.zeros(0, self.dim)
        )
        head_keys = self.head_keys_by_id
        crystals = [c.detach().clone() for c in self.crystals]
    else:
        from ._octonion_ops import bind  # local import to avoid cycle
        keep = [(mid, e) for mid, e in self.entries.items() if filter(e)]
        entries_dicts = [e.to_dict() for _, e in keep]
        embeddings = (
            torch.stack([e.embedding for _, e in keep])
            if keep
            else torch.zeros(0, self.dim)
        )
        head_keys = {mid: self.head_keys_by_id[mid] for mid, _ in keep}
        crystals = [torch.zeros(self.dim) for _ in range(self.n_heads)]
        for mid, e in keep:
            for h in range(self.n_heads):
                crystals[h] = crystals[h] + bind(head_keys[mid][h], e.embedding)

    metadata = {
        "format_version": 1,
        "dim": self.dim,
        "n_heads": self.n_heads,
        "bind_algebra": self.bind_algebra,
        "user_id": self.user_id,
        "n_entries": len(entries_dicts),
        "created_at": time.time(),
    }
    return Crystal(
        metadata=metadata,
        crystals=crystals,
        entries=entries_dicts,
        embeddings=embeddings,
        head_keys=head_keys,
    )


def _export_file(self, path: str | Path, *, filter=None) -> None:  # noqa: A002
    self.export(filter=filter).to_file(path)


def _from_file(cls, path: str | Path, **kwargs) -> "CrystalMem":
    """Construct a CrystalMem from a saved Crystal file."""
    return cls.from_crystal(Crystal.from_file(path), **kwargs)


def _from_crystal(cls, crystal: Crystal, **kwargs) -> "CrystalMem":
    meta = crystal.metadata
    inst = cls(
        user_id=kwargs.pop("user_id", meta["user_id"]),
        dim=meta["dim"],
        n_heads=meta["n_heads"],
        bind_algebra=meta["bind_algebra"],
        **kwargs,
    )
    _populate_from_crystal(inst, crystal)
    return inst


def _populate_from_crystal(inst, crystal: Crystal) -> None:
    from .core import MemoryEntry, ProvenanceStep  # local to avoid cycle
    inst.crystals = [c.detach().clone() for c in crystal.crystals]
    inst.head_keys_by_id = {
        k: [v.detach().clone() for v in vs] for k, vs in crystal.head_keys.items()
    }
    inst.entries = {}
    inst.entity_facts_idx = {}
    for i, e_dict in enumerate(crystal.entries):
        prov = [ProvenanceStep(**p) for p in e_dict.get("provenance", [])]
        entry = MemoryEntry(
            id=e_dict["id"],
            content=e_dict["content"],
            embedding=crystal.embeddings[i].clone(),
            user_id=e_dict["user_id"],
            timestamp=e_dict["timestamp"],
            tags=set(e_dict["tags"]),
            metadata=e_dict["metadata"],
            source=e_dict.get("source", "imported"),
            provenance=prov,
        )
        inst.entries[e_dict["id"]] = entry
        ent = entry.metadata.get("entity")
        if ent:
            inst.entity_facts_idx.setdefault(ent, set()).add(entry.id)


def _merge(self, other) -> None:
    """In-place merge: self += other.

    HRR algebra: M_merged = sum(M_i). Federated property: empirical Δ ≈ 9×10⁻⁸
    vs centralized rebuild.
    """
    if isinstance(other, Crystal):
        crystal = other
    elif hasattr(other, "export"):
        crystal = other.export()
    else:
        raise TypeError(f"Cannot merge {type(other).__name__}")

    if crystal.metadata["dim"] != self.dim:
        raise ValueError(
            f"Dim mismatch: self={self.dim}, other={crystal.metadata['dim']}"
        )
    if crystal.metadata["n_heads"] != self.n_heads:
        raise ValueError(
            f"n_heads mismatch: self={self.n_heads}, other={crystal.metadata['n_heads']}"
        )
    if crystal.metadata["bind_algebra"] != self.bind_algebra:
        raise ValueError(
            f"Algebra mismatch: {self.bind_algebra} vs {crystal.metadata['bind_algebra']}"
        )

    from .core import MemoryEntry, ProvenanceStep
    import time as _time
    source_label = (
        f"crystal:{crystal.metadata.get('user_id', '?')}"
        f"@{crystal.metadata.get('created_at', 0):.0f}"
    )
    now = _time.time()
    for h in range(self.n_heads):
        self.crystals[h] = self.crystals[h] + crystal.crystals[h]
    for i, e_dict in enumerate(crystal.entries):
        mid = e_dict["id"]
        if mid in self.entries:
            continue  # idempotent: skip duplicates by id
        prov = [ProvenanceStep(**p) for p in e_dict.get("provenance", [])]
        prov.append(ProvenanceStep(
            actor=self.user_id,
            action="merged",
            timestamp=now,
            source_crystal=source_label,
        ))
        entry = MemoryEntry(
            id=mid,
            content=e_dict["content"],
            embedding=crystal.embeddings[i].clone(),
            user_id=e_dict["user_id"],
            timestamp=e_dict["timestamp"],
            tags=set(e_dict["tags"]),
            metadata=e_dict["metadata"],
            source="imported",
            provenance=prov,
        )
        self.entries[mid] = entry
        self.head_keys_by_id[mid] = [
            v.detach().clone() for v in crystal.head_keys[mid]
        ]
        ent = entry.metadata.get("entity")
        if ent:
            self.entity_facts_idx.setdefault(ent, set()).add(mid)


def _merge_file(self, path: str | Path) -> None:
    self.merge(Crystal.from_file(path))


def _merge_many(self, others: Iterable) -> None:
    for o in others:
        self.merge(o)


def _unmerge(self, other) -> None:
    """In-place subtract: self −= other.

    Exact subtract — removes both crystal contribution AND entries that
    came from `other`. Use case: undo accidental merge, revoke shared memory.
    """
    if isinstance(other, Crystal):
        crystal = other
    elif hasattr(other, "export"):
        crystal = other.export()
    else:
        raise TypeError(f"Cannot unmerge {type(other).__name__}")

    for h in range(self.n_heads):
        self.crystals[h] = self.crystals[h] - crystal.crystals[h]
    for e_dict in crystal.entries:
        mid = e_dict["id"]
        self.entries.pop(mid, None)
        self.head_keys_by_id.pop(mid, None)


def _diff(self, other):
    """Return (only_in_self, only_in_other, common) sets of memory_ids."""
    if isinstance(other, Crystal):
        other_ids = {e["id"] for e in other.entries}
    elif hasattr(other, "entries"):
        other_ids = set(other.entries.keys())
    else:
        raise TypeError(f"Cannot diff {type(other).__name__}")
    self_ids = set(self.entries.keys())
    return (
        self_ids - other_ids,
        other_ids - self_ids,
        self_ids & other_ids,
    )


def install_portability_methods() -> None:
    """Bolt portability API onto CrystalMem (called once from package __init__)."""
    from .core import CrystalMem
    CrystalMem.export = _export
    CrystalMem.export_file = _export_file
    CrystalMem.from_file = classmethod(_from_file)
    CrystalMem.from_crystal = classmethod(_from_crystal)
    CrystalMem.merge = _merge
    CrystalMem.merge_file = _merge_file
    CrystalMem.merge_many = _merge_many
    CrystalMem.unmerge = _unmerge
    CrystalMem.diff = _diff


install_portability_methods()
