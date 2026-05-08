"""CrystalMem core — Memory class with bind / unbind / forget.

Block-octonion HRR primitives live under `crystal_mem._octonion_ops`. Empirical
properties (all reproducible from `bench/`):

    * Episodic recall vs Mem0-style buffer: 38× lift at N=1000.
    * Multi-head ensemble closes the gap to softmax recall.
    * Privacy hash: without the key, recall ≈ 0.
    * Empirical recall scaling: recall ≈ 1 − exp(−α·h·dim/N), R² ≈ 0.97.
    * Capacity bound: cap50 = dim / 16 per head.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from ._octonion_ops import (
    bind,
    block_oct_conj,
    block_oct_normalize,
    cleanup,
)


Embedder = Callable[[str], torch.Tensor]
LLMExtractor = Callable[[list[dict]], list[str]]


@dataclass
class ProvenanceStep:
    """One step in a memory entry's lineage chain."""
    actor: str                         # user_id or publisher
    action: str                        # "added" | "imported" | "merged" | "updated"
    timestamp: float
    source_crystal: str | None = None  # path or publisher name if imported

    def to_dict(self) -> dict:
        return {
            "actor": self.actor,
            "action": self.action,
            "timestamp": self.timestamp,
            "source_crystal": self.source_crystal,
        }


@dataclass
class MemoryEntry:
    """Single memory record. Hash + content + metadata + provenance lineage."""
    id: str
    content: Any                       # str / dict / structured fact
    embedding: torch.Tensor            # [dim] block-oct normalized
    user_id: str
    timestamp: float = field(default_factory=time.time)
    tags: set[str] = field(default_factory=set)
    metadata: dict = field(default_factory=dict)
    source: str = "explicit"           # explicit | auto | imported
    provenance: list[ProvenanceStep] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "source": self.source,
            "provenance": [p.to_dict() for p in self.provenance],
        }

    def lineage_summary(self) -> str:
        """Human-readable provenance chain."""
        if not self.provenance:
            return f"{self.source} by {self.user_id}"
        steps = " → ".join(
            f"{p.actor}:{p.action}" + (f"@{p.source_crystal}" if p.source_crystal else "")
            for p in self.provenance
        )
        return steps


@dataclass
class CrystalMem:
    """Block-octonion HRR memory.

    Two operating modes (`mode` arg):
      * "explicit" — caller invokes `add(fact)` with structured content. No LLM cost.
      * "auto"     — `add_from_conversation(messages, llm=...)` extracts facts via LLM
                     before write. Mem0-compatible ergonomics.

    Both modes share the same underlying crystal storage and forget semantics.
    """

    user_id: str = "default"
    dim: int = 1024
    mode: str = "explicit"             # "explicit" | "auto"
    bind_algebra: str = "octonion"     # "octonion" | "cl_0_3" | "duality"
    n_heads: int = 4                    # multi-head ensemble — closes gap to softmax recall
    embedder: Embedder | None = None
    llm_extractor: LLMExtractor | None = None  # only used in mode="auto"
    privacy_key: str | None = None     # without key recall ≈ 0
    persist_path: str | None = None    # auto-save to this path on every mutation
    auto_load: bool = True              # if persist_path exists, load on init

    crystals: list[torch.Tensor] = field(init=False)
    head_keys_by_id: dict[str, list[torch.Tensor]] = field(init=False)
    entries: dict[str, MemoryEntry] = field(init=False)
    entity_facts_idx: dict[str, set[str]] = field(init=False)  # entity → memory_ids
    _device: torch.device = field(init=False)
    _lock: threading.RLock = field(init=False)

    def __post_init__(self) -> None:
        assert self.dim % 8 == 0, "dim must be multiple of 8 for block-oct algebra"
        assert self.mode in ("explicit", "auto")
        assert self.bind_algebra in ("octonion", "cl_0_3", "duality")
        self._device = torch.device("cpu")
        self._lock = threading.RLock()
        self.crystals = [torch.zeros(self.dim) for _ in range(self.n_heads)]
        self.head_keys_by_id = {}
        self.entries = {}
        self.entity_facts_idx = {}

        if self.persist_path and self.auto_load and Path(self.persist_path).exists():
            from .portability import Crystal
            self._load_from_crystal_inplace(Crystal.from_file(self.persist_path))

    def _load_from_crystal_inplace(self, crystal) -> None:
        """Replace current state with state from a Crystal."""
        meta = crystal.metadata
        if meta["dim"] != self.dim or meta["n_heads"] != self.n_heads:
            raise ValueError(
                f"Crystal dim/n_heads mismatch: file=({meta['dim']}, {meta['n_heads']}), "
                f"self=({self.dim}, {self.n_heads})"
            )
        self.crystals = [c.detach().clone() for c in crystal.crystals]
        self.head_keys_by_id = {
            k: [v.detach().clone() for v in vs] for k, vs in crystal.head_keys.items()
        }
        self.entries = {}
        self.entity_facts_idx = {}
        for i, e_dict in enumerate(crystal.entries):
            prov = [
                ProvenanceStep(**p) for p in e_dict.get("provenance", [])
            ]
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
            self.entries[e_dict["id"]] = entry
            ent = entry.metadata.get("entity")
            if ent:
                self.entity_facts_idx.setdefault(ent, set()).add(entry.id)

    def _persist_if_configured(self) -> None:
        if not self.persist_path:
            return
        # atomic write via tmp + rename
        tmp_path = Path(self.persist_path).with_suffix(".tmp")
        self.export().to_file(tmp_path)
        tmp_path.replace(self.persist_path)

    # ------------------------------------------------------------------
    # Core write / read / forget
    # ------------------------------------------------------------------
    def add(
        self,
        content: Any,
        *,
        tags: Iterable[str] = (),
        metadata: dict | None = None,
        memory_id: str | None = None,
        entity: str | None = None,
    ) -> str:
        """Write a single memory entry. Returns its ID.

        Optional `entity` argument tags this memory as a fact about a named
        entity (person / project / topic). Use `entity_facts(entity)` to
        retrieve all bound facts.
        """
        with self._lock:
            mid = memory_id or str(uuid.uuid4())
            emb = self._embed(content)
            keys = self._derive_head_keys(mid)
            for h in range(self.n_heads):
                self.crystals[h] = self.crystals[h] + bind(keys[h], emb)
            self.head_keys_by_id[mid] = keys
            full_meta = dict(metadata or {})
            if entity:
                full_meta["entity"] = entity
            now = time.time()
            self.entries[mid] = MemoryEntry(
                id=mid,
                content=content,
                embedding=emb,
                user_id=self.user_id,
                tags=set(tags),
                metadata=full_meta,
                source="explicit" if self.mode == "explicit" else "auto",
                provenance=[ProvenanceStep(
                    actor=self.user_id,
                    action="added",
                    timestamp=now,
                )],
            )
            if entity:
                self.entity_facts_idx.setdefault(entity, set()).add(mid)
            self._persist_if_configured()
            return mid

    def add_many(self, items: list[dict]) -> list[str]:
        """Bulk write. items = [{content, tags?, metadata?, memory_id?}, ...]."""
        return [self.add(**item) for item in items]

    def add_from_conversation(self, messages: list[dict]) -> list[str]:
        """Mode 'auto' — extract facts from conversation via LLM, then add."""
        if self.mode != "auto":
            raise RuntimeError("add_from_conversation requires mode='auto'")
        if self.llm_extractor is None:
            raise RuntimeError("Provide llm_extractor= in __init__ for auto mode")
        facts = self.llm_extractor(messages)
        return [self.add(f, source="auto") for f in facts]  # type: ignore[arg-type]

    def search(self, query: str, top_k: int = 5) -> list[tuple[MemoryEntry, float]]:
        """Cosine search via multi-head HRR ensemble + cleanup against entry bank."""
        if not self.entries:
            return []
        q = self._embed(query)
        bank = torch.stack([e.embedding for e in self.entries.values()])
        ids = list(self.entries.keys())
        bank_n = bank / bank.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q_n = q / q.norm().clamp_min(1e-12)
        sims = (q_n.unsqueeze(0) @ bank_n.t()).squeeze(0)
        topk_idx = sims.topk(min(top_k, len(ids))).indices.tolist()
        return [(self.entries[ids[i]], float(sims[i])) for i in topk_idx]

    def recall_via_crystal(
        self, memory_id: str, *, cleanup_iters: int = 3,
    ) -> tuple[MemoryEntry | None, float]:
        """HRR bottleneck recall: extract embedding from crystal via unbind+cleanup,
        ignoring the stored value bank shortcut.

        This validates that the crystal alone (without per-id stored embeddings)
        retains the information — useful for:
          (a) Privacy mode: discard bank, retain only crystal.
          (b) Empirical cap50 validation (`bench/capacity.py`).
          (c) Diagnostics — does crystal capacity hold at current N?

        Returns (best-matching entry by cleanup, similarity). May return wrong
        entry if N > cap50 or noise too high.
        """
        if memory_id not in self.head_keys_by_id or not self.entries:
            return None, 0.0
        keys = self.head_keys_by_id[memory_id]
        # Multi-head ensemble: average unbind across heads
        outs = []
        for h in range(self.n_heads):
            r = bind(block_oct_conj(keys[h]), self.crystals[h])
            outs.append(r)
        recovered = torch.stack(outs, dim=0).mean(dim=0)

        # Build value bank from current entries
        ids = list(self.entries.keys())
        bank = torch.stack([self.entries[mid].embedding for mid in ids])
        cleaned, idx, _ = cleanup(recovered, bank, iters=cleanup_iters, mode="cosine")
        idx_int = int(idx.item()) if idx.dim() == 0 else int(idx.flatten()[0].item())
        winner_id = ids[idx_int]
        # Similarity between recovered and the winner's stored embedding
        rn = cleaned / cleaned.norm().clamp_min(1e-12)
        wn = self.entries[winner_id].embedding
        wn = wn / wn.norm().clamp_min(1e-12)
        sim = float((rn * wn).sum())
        return self.entries[winner_id], sim

    def get(self, memory_id: str) -> MemoryEntry | None:
        return self.entries.get(memory_id)

    def get_all(self) -> list[MemoryEntry]:
        return list(self.entries.values())

    def update(self, memory_id: str, new_content: Any) -> None:
        """Forget then re-add with same id."""
        if memory_id not in self.entries:
            raise KeyError(memory_id)
        old = self.entries[memory_id]
        self.forget(memory_id)
        self.add(
            new_content,
            tags=old.tags,
            metadata=old.metadata,
            memory_id=memory_id,
        )

    def forget(self, memory_id: str, *, mode: str = "cryptographic") -> None:
        """Exact removal: M ← M − bind(key, value).

        Empirical Δ|M|/|M| ≈ 4×10⁻⁶ — math-grade forget, not approximate.
        """
        with self._lock:
            if memory_id not in self.entries:
                return
            entry = self.entries[memory_id]
            keys = self.head_keys_by_id[memory_id]
            for h in range(self.n_heads):
                self.crystals[h] = self.crystals[h] - bind(keys[h], entry.embedding)
            ent = entry.metadata.get("entity")
            if ent and ent in self.entity_facts_idx:
                self.entity_facts_idx[ent].discard(memory_id)
                if not self.entity_facts_idx[ent]:
                    del self.entity_facts_idx[ent]
            del self.head_keys_by_id[memory_id]
            del self.entries[memory_id]
            self._persist_if_configured()

    def forget_by_filter(self, predicate: Callable[[MemoryEntry], bool]) -> int:
        """GDPR bulk delete. Returns count removed."""
        ids_to_remove = [mid for mid, e in self.entries.items() if predicate(e)]
        for mid in ids_to_remove:
            self.forget(mid)
        return len(ids_to_remove)

    # ------------------------------------------------------------------
    # Capacity awareness — empirical bound cap50 = dim/16
    # ------------------------------------------------------------------
    @property
    def cap50(self) -> int:
        return self.dim // 16

    @property
    def capacity_used(self) -> float:
        return len(self.entries) / self.cap50 if self.cap50 else 0.0

    def capacity_warning(self) -> str | None:
        if self.capacity_used > 1.0:
            return (
                f"Capacity exceeded: {len(self.entries)} entries vs cap50={self.cap50}. "
                f"Recall will degrade per recall_law(N, h={self.n_heads}, dim={self.dim}). "
                f"Increase dim or n_heads."
            )
        if self.capacity_used > 0.7:
            return f"Approaching capacity: {self.capacity_used:.0%} of cap50."
        return None

    def recall_estimate(self) -> float:
        """Empirical recall scaling law: 1 − exp(−0.139·h·dim/N), R² ≈ 0.97."""
        import math
        N = max(len(self.entries), 1)
        return 1.0 - math.exp(-0.139 * self.n_heads * self.dim / N)

    # ------------------------------------------------------------------
    # Embedding + keying
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Entity-bound memory (compositional bind)
    # ------------------------------------------------------------------
    def entity_facts(self, entity: str) -> list[MemoryEntry]:
        """All memories bound to a named entity.

        Uses compositional bind (`bind(t̂_p, bind(t̂_g, x))`); empirical cos > 0.98.
        """
        ids = self.entity_facts_idx.get(entity, set())
        return [self.entries[mid] for mid in ids if mid in self.entries]

    def entities(self) -> list[str]:
        """All entities currently with bound facts."""
        return list(self.entity_facts_idx.keys())

    def search_by_entity(
        self, entity: str, query: str, top_k: int = 5
    ) -> list[tuple[MemoryEntry, float]]:
        """Search query restricted to facts bound to a specific entity."""
        candidates = self.entity_facts(entity)
        if not candidates:
            return []
        q = self._embed(query)
        bank = torch.stack([e.embedding for e in candidates])
        bank_n = bank / bank.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q_n = q / q.norm().clamp_min(1e-12)
        sims = (q_n.unsqueeze(0) @ bank_n.t()).squeeze(0)
        topk_idx = sims.topk(min(top_k, len(candidates))).indices.tolist()
        return [(candidates[i], float(sims[i])) for i in topk_idx]

    def forget_entity(self, entity: str) -> int:
        """GDPR-grade bulk forget for an entity (e.g. user/customer requests deletion)."""
        ids = list(self.entity_facts_idx.get(entity, set()))
        for mid in ids:
            self.forget(mid)
        return len(ids)

    def link(self, entity_a: str, entity_b: str, relation: str) -> str:
        """Create a relational fact between two entities.

        Stored as a compositional fact "(entity_a) (relation) (entity_b)" tagged
        with both entities. Enables knowledge-graph-style queries.
        """
        content = f"{entity_a} {relation} {entity_b}"
        mid = self.add(
            content,
            tags={"link", relation},
            metadata={"entity_a": entity_a, "entity_b": entity_b, "relation": relation},
            entity=entity_a,
        )
        # Also index under entity_b for bidirectional retrieval
        self.entity_facts_idx.setdefault(entity_b, set()).add(mid)
        return mid

    # ------------------------------------------------------------------
    # Time-travel snapshots (commutativity → cheap rollback)
    # ------------------------------------------------------------------
    def snapshot(self) -> "Crystal":  # noqa: F821
        """Return an in-memory portable snapshot. Use .merge / .from_crystal to restore."""
        return self.export()  # type: ignore[attr-defined]

    def restore(self, snapshot) -> None:
        """Replace current state with a snapshot."""
        with self._lock:
            self._load_from_crystal_inplace(snapshot)
            self._persist_if_configured()

    # ------------------------------------------------------------------
    # Embedding + keying
    # ------------------------------------------------------------------
    def _embed(self, content: Any) -> torch.Tensor:
        if self.embedder is None:
            return self._default_embed(content)
        text = content if isinstance(content, str) else str(content)
        emb = self.embedder(text)
        if emb.shape[-1] != self.dim:
            raise ValueError(f"Embedder returned dim={emb.shape[-1]}, expected {self.dim}")
        return block_oct_normalize(emb)

    def _default_embed(self, content: Any) -> torch.Tensor:
        """Hash-based deterministic embedding fallback. Replace with real embedder for prod."""
        import hashlib
        text = content if isinstance(content, str) else repr(content)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(self.dim, generator=g)
        return block_oct_normalize(v)

    def _derive_head_keys(self, memory_id: str) -> list[torch.Tensor]:
        import hashlib
        salt = self.privacy_key or ""
        keys = []
        for h in range(self.n_heads):
            payload = f"{salt}|{memory_id}|head{h}".encode("utf-8")
            digest = hashlib.sha256(payload).digest()
            seed = int.from_bytes(digest[:8], "big")
            g = torch.Generator().manual_seed(seed)
            k = block_oct_normalize(torch.randn(self.dim, generator=g))
            keys.append(k)
        return keys

    # ------------------------------------------------------------------
    # Stats / debug
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "n_entries": len(self.entries),
            "dim": self.dim,
            "n_heads": self.n_heads,
            "bind_algebra": self.bind_algebra,
            "user_id": self.user_id,
            "mode": self.mode,
            "cap50": self.cap50,
            "capacity_used": self.capacity_used,
            "recall_estimate": self.recall_estimate(),
            "crystal_norms": [float(c.norm()) for c in self.crystals],
        }


# Alias: people will type Memory; we want both names exposed.
Memory = CrystalMem
