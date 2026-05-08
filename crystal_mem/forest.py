"""CrystalForest — bank of crystals with auto-placement, compaction, KG hybrid.

The core problem with single CrystalMem at scale:
    cap50 = dim/16 per head. With dim=1024, h=4 ensemble holds clean recall up
    to ~256 entries via HRR-bottleneck. Bank-mode (cosine over stored embeddings)
    has no such limit but loses the algebraic property — it's just FAISS.

The solution: many small crystals, each held below cap50, with a router that
decides where to write/read. Auto-clustering (k-means on embeddings) keeps
each crystal semantically coherent. Auto-compaction splits/merges crystals
based on capacity utilization.

Layered with a knowledge graph:
    nodes  = entities (people, projects, topics)
    edges  = relations between entities
    each node owns a small crystal of facts about that entity
    each edge owns a small crystal of relation context

Combines hierarchical memory routing with knowledge-graph overlay in one
self-contained module.

Public API:
    CrystalForest(dim, target_per_crystal, max_per_crystal)
        .add(content, entity=...)       — auto-route to right crystal
        .search(query, top_k)            — search across forest
        .search_entity(entity, query)    — KG-routed search
        .compact()                       — split overfull, merge underfull
        .stats()                         — per-shard stats
        .export_file(path)               — full forest as one .crystal file
        .merge_forest(other)             — merge two forests

Status: experimental — intended as Layer-0 INFRASTRUCTURE for the product
layers in USE_CASES.md. Probably NOT a standalone product.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch

from ._octonion_ops import block_oct_normalize


@dataclass
class Shard:
    """One crystal in the forest. Wraps a CrystalMem as a sub-storage."""
    shard_id: str
    centroid: torch.Tensor                      # representative vector
    memory: "CrystalMem"  # noqa: F821
    last_compacted: float = field(default_factory=time.time)


@dataclass
class KGNode:
    """Knowledge-graph node for an entity."""
    entity: str
    crystal_centroid: torch.Tensor              # canonical embedding for entity
    fact_shard_ids: set[str] = field(default_factory=set)
    edges_in: set[str] = field(default_factory=set)
    edges_out: set[str] = field(default_factory=set)


@dataclass
class KGEdge:
    """Knowledge-graph edge with relation context."""
    edge_id: str
    src: str
    dst: str
    relation: str
    shard_id: str | None = None                 # crystal storing context


class CrystalForest:
    """A bank of crystals + KG + auto-compaction."""

    def __init__(
        self,
        dim: int = 1024,
        target_per_crystal: int = 64,
        max_per_crystal: int = 256,
        n_heads: int = 4,
        embedder: Callable[[str], torch.Tensor] | None = None,
        bind_algebra: str = "octonion",
        user_id: str = "default",
        privacy_key: str | None = None,
    ):
        if dim % 8 != 0:
            raise ValueError("dim must be multiple of 8")
        self.dim = dim
        self.target_per_crystal = target_per_crystal
        self.max_per_crystal = max_per_crystal
        self.n_heads = n_heads
        self.embedder = embedder
        self.bind_algebra = bind_algebra
        self.user_id = user_id
        self.privacy_key = privacy_key

        self.shards: dict[str, Shard] = {}
        self.kg_nodes: dict[str, KGNode] = {}
        self.kg_edges: dict[str, KGEdge] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Internal: shard creation + routing
    # ------------------------------------------------------------------
    def _new_shard(self, centroid: torch.Tensor | None = None) -> Shard:
        from .core import CrystalMem
        shard_id = f"s_{len(self.shards):04d}_{uuid.uuid4().hex[:6]}"
        if centroid is None:
            centroid = torch.zeros(self.dim)
        mem = CrystalMem(
            user_id=self.user_id, dim=self.dim, n_heads=self.n_heads,
            embedder=self.embedder, bind_algebra=self.bind_algebra,
            privacy_key=self.privacy_key,
        )
        s = Shard(shard_id=shard_id, centroid=centroid.clone(), memory=mem)
        self.shards[shard_id] = s
        return s

    def _embed(self, content: Any) -> torch.Tensor:
        if self.embedder is None:
            import hashlib
            text = content if isinstance(content, str) else repr(content)
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big")
            g = torch.Generator().manual_seed(seed)
            return block_oct_normalize(torch.randn(self.dim, generator=g))
        text = content if isinstance(content, str) else str(content)
        return block_oct_normalize(self.embedder(text))

    def _route_for_write(self, emb: torch.Tensor) -> Shard:
        """Pick best shard for a new entry. Create new if none under target."""
        if not self.shards:
            return self._new_shard(centroid=emb)
        emb_n = emb / emb.norm().clamp_min(1e-12)

        # Score each shard by (centroid similarity, capacity available)
        candidates = []
        for s in self.shards.values():
            if len(s.memory.entries) >= self.max_per_crystal:
                continue                        # skip full shards
            c_n = s.centroid / s.centroid.norm().clamp_min(1e-12)
            sim = float((emb_n * c_n).sum())
            cap_score = 1.0 - (len(s.memory.entries) / self.max_per_crystal)
            score = sim * 0.7 + cap_score * 0.3
            candidates.append((score, s))

        if not candidates:
            return self._new_shard(centroid=emb)

        candidates.sort(key=lambda x: -x[0])
        best_score, best_shard = candidates[0]

        # If best similarity below threshold AND we don't already have many
        # below-target shards, spawn a new shard for this distinct semantic cluster.
        below_target = sum(
            1 for s in self.shards.values()
            if len(s.memory.entries) < self.target_per_crystal
        )
        if best_score < 0.15 and below_target == 0:
            return self._new_shard(centroid=emb)

        return best_shard

    def _update_centroid(self, shard: Shard) -> None:
        """Recompute centroid as mean of stored embeddings."""
        if not shard.memory.entries:
            return
        embs = torch.stack([e.embedding for e in shard.memory.entries.values()])
        shard.centroid = embs.mean(dim=0)

    # ------------------------------------------------------------------
    # Public: add / search
    # ------------------------------------------------------------------
    def add(
        self,
        content: Any,
        *,
        entity: str | None = None,
        tags: Iterable[str] = (),
        metadata: dict | None = None,
    ) -> tuple[str, str]:
        """Write entry. Returns (shard_id, memory_id)."""
        with self._lock:
            emb = self._embed(content)
            shard = self._route_for_write(emb)
            mid = shard.memory.add(
                content,
                tags=list(tags) + [f"shard:{shard.shard_id}"],
                metadata=metadata or {},
                entity=entity,
            )
            # incremental centroid update (running mean)
            n = len(shard.memory.entries)
            shard.centroid = (shard.centroid * (n - 1) + emb) / n

            # KG node for entity
            if entity:
                if entity not in self.kg_nodes:
                    self.kg_nodes[entity] = KGNode(
                        entity=entity,
                        crystal_centroid=emb.clone(),
                    )
                self.kg_nodes[entity].fact_shard_ids.add(shard.shard_id)

            # auto-compact if shard near capacity
            if len(shard.memory.entries) >= self.max_per_crystal:
                self._split_shard(shard)

            return shard.shard_id, mid

    def link(self, src: str, dst: str, relation: str,
             context: str | None = None) -> str:
        """Create a KG edge. Optionally store context fact in the edge's crystal."""
        with self._lock:
            for ent in (src, dst):
                if ent not in self.kg_nodes:
                    self.kg_nodes[ent] = KGNode(
                        entity=ent,
                        crystal_centroid=self._embed(ent),
                    )
            edge_id = f"e_{src}__{relation}__{dst}_{uuid.uuid4().hex[:6]}"
            edge = KGEdge(edge_id=edge_id, src=src, dst=dst, relation=relation)
            if context:
                shard_id, _ = self.add(
                    context,
                    entity=src,
                    tags=["edge", relation],
                    metadata={"edge_id": edge_id, "src": src, "dst": dst,
                             "relation": relation},
                )
                edge.shard_id = shard_id
            self.kg_edges[edge_id] = edge
            self.kg_nodes[src].edges_out.add(edge_id)
            self.kg_nodes[dst].edges_in.add(edge_id)
            return edge_id

    def search(self, query: str, top_k: int = 5,
               max_shards_to_probe: int = 8) -> list[tuple[str, str, Any, float]]:
        """Search the forest. Returns list of (shard_id, mem_id, content, score)."""
        if not self.shards:
            return []
        q = self._embed(query)
        qn = q / q.norm().clamp_min(1e-12)

        # Pick top-K shards by centroid similarity
        scored = []
        for s in self.shards.values():
            if not s.memory.entries:
                continue
            cn = s.centroid / s.centroid.norm().clamp_min(1e-12)
            scored.append((float((qn * cn).sum()), s))
        scored.sort(key=lambda x: -x[0])
        probed = scored[:max_shards_to_probe]

        # Search within each probed shard, gather all hits
        hits = []
        for _, shard in probed:
            results = shard.memory.search(query, top_k=top_k)
            for entry, score in results:
                hits.append((shard.shard_id, entry.id, entry.content, score))

        hits.sort(key=lambda h: -h[3])
        return hits[:top_k]

    def search_entity(self, entity: str, query: str,
                      top_k: int = 5) -> list[tuple[str, Any, float]]:
        """KG-routed search: only shards holding facts about `entity`."""
        if entity not in self.kg_nodes:
            return []
        node = self.kg_nodes[entity]
        hits = []
        for sid in node.fact_shard_ids:
            shard = self.shards.get(sid)
            if shard is None or not shard.memory.entries:
                continue
            results = shard.memory.search_by_entity(entity, query, top_k=top_k)
            for entry, score in results:
                hits.append((sid, entry.content, score))
        hits.sort(key=lambda h: -h[2])
        return hits[:top_k]

    def neighbors(self, entity: str, depth: int = 1) -> list[str]:
        """KG neighbors of entity up to depth."""
        if entity not in self.kg_nodes:
            return []
        seen = {entity}
        frontier = {entity}
        for _ in range(depth):
            new_frontier = set()
            for ent in frontier:
                node = self.kg_nodes.get(ent)
                if not node:
                    continue
                for eid in node.edges_out | node.edges_in:
                    e = self.kg_edges[eid]
                    other = e.dst if e.src == ent else e.src
                    if other not in seen:
                        new_frontier.add(other)
                        seen.add(other)
            frontier = new_frontier
            if not frontier:
                break
        return [e for e in seen if e != entity]

    # ------------------------------------------------------------------
    # Auto-compaction
    # ------------------------------------------------------------------
    def _split_shard(self, shard: Shard) -> list[Shard]:
        """Split overfull shard into 2 via k-means on embeddings."""
        if len(shard.memory.entries) < 4:
            return [shard]

        ids = list(shard.memory.entries.keys())
        embs = torch.stack([shard.memory.entries[i].embedding for i in ids])
        # 2-means with random init
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(ids), generator=g)[:2]
        c0 = embs[idx[0]].clone()
        c1 = embs[idx[1]].clone()
        for _ in range(5):
            c0n = c0 / c0.norm().clamp_min(1e-12)
            c1n = c1 / c1.norm().clamp_min(1e-12)
            embs_n = embs / embs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            sim0 = embs_n @ c0n
            sim1 = embs_n @ c1n
            mask = sim0 >= sim1
            if mask.sum() == 0 or mask.sum() == len(ids):
                break
            c0 = embs[mask].mean(dim=0)
            c1 = embs[~mask].mean(dim=0)

        shard0 = self._new_shard(centroid=c0)
        shard1 = self._new_shard(centroid=c1)

        for i, mid in enumerate(ids):
            entry = shard.memory.entries[mid]
            target = shard0 if mask[i] else shard1
            target.memory.add(
                entry.content,
                tags=entry.tags - {f"shard:{shard.shard_id}"},
                metadata=entry.metadata,
                entity=entry.metadata.get("entity"),
                memory_id=mid,
            )
            # Rewire KG node
            ent = entry.metadata.get("entity")
            if ent and ent in self.kg_nodes:
                self.kg_nodes[ent].fact_shard_ids.discard(shard.shard_id)
                self.kg_nodes[ent].fact_shard_ids.add(target.shard_id)

        del self.shards[shard.shard_id]
        return [shard0, shard1]

    def _maybe_merge_pair(self, s_a: Shard, s_b: Shard) -> bool:
        """Merge two underfull, semantically-similar shards."""
        if (len(s_a.memory.entries) + len(s_b.memory.entries)
                > self.max_per_crystal):
            return False
        a_n = s_a.centroid / s_a.centroid.norm().clamp_min(1e-12)
        b_n = s_b.centroid / s_b.centroid.norm().clamp_min(1e-12)
        sim = float((a_n * b_n).sum())
        if sim < 0.5:                           # not similar enough
            return False

        # Move all entries from B into A
        for mid, entry in list(s_b.memory.entries.items()):
            s_a.memory.add(
                entry.content,
                tags=entry.tags - {f"shard:{s_b.shard_id}"},
                metadata=entry.metadata,
                entity=entry.metadata.get("entity"),
                memory_id=mid,
            )
            ent = entry.metadata.get("entity")
            if ent and ent in self.kg_nodes:
                self.kg_nodes[ent].fact_shard_ids.discard(s_b.shard_id)
                self.kg_nodes[ent].fact_shard_ids.add(s_a.shard_id)

        # Re-fit centroid
        self._update_centroid(s_a)
        del self.shards[s_b.shard_id]
        return True

    def compact(self) -> dict:
        """Force a compaction pass: split overfull, merge similar underfull."""
        with self._lock:
            initial = len(self.shards)

            # Split overfull
            for sid, s in list(self.shards.items()):
                if len(s.memory.entries) >= self.max_per_crystal:
                    self._split_shard(s)

            # Merge underfull pairs
            underfull = [s for s in self.shards.values()
                         if len(s.memory.entries) < self.target_per_crystal // 4]
            merged = 0
            for i, s_a in enumerate(underfull):
                for s_b in underfull[i + 1:]:
                    if (s_a.shard_id in self.shards and s_b.shard_id in self.shards
                            and self._maybe_merge_pair(s_a, s_b)):
                        merged += 1

            return {
                "initial_shards": initial,
                "final_shards": len(self.shards),
                "merged_pairs": merged,
            }

    # ------------------------------------------------------------------
    # Stats / persistence
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        total_entries = sum(len(s.memory.entries) for s in self.shards.values())
        sizes = [len(s.memory.entries) for s in self.shards.values()]
        return {
            "n_shards": len(self.shards),
            "total_entries": total_entries,
            "n_kg_nodes": len(self.kg_nodes),
            "n_kg_edges": len(self.kg_edges),
            "shard_size_min": min(sizes) if sizes else 0,
            "shard_size_max": max(sizes) if sizes else 0,
            "shard_size_mean": (sum(sizes) / len(sizes)) if sizes else 0,
            "avg_capacity_used": (
                sum(s.memory.capacity_used for s in self.shards.values())
                / max(len(self.shards), 1)
            ),
        }

    def export_file(self, path: str) -> None:
        """Serialize entire forest. Each shard is one Crystal in a manifest dict."""
        from pathlib import Path
        import torch as _t
        manifest = {
            "format_version": 1,
            "dim": self.dim,
            "target_per_crystal": self.target_per_crystal,
            "max_per_crystal": self.max_per_crystal,
            "n_heads": self.n_heads,
            "bind_algebra": self.bind_algebra,
            "shards": [],
            "kg_nodes": {
                ent: {
                    "fact_shard_ids": list(n.fact_shard_ids),
                    "edges_in": list(n.edges_in),
                    "edges_out": list(n.edges_out),
                }
                for ent, n in self.kg_nodes.items()
            },
            "kg_edges": {
                eid: {"src": e.src, "dst": e.dst, "relation": e.relation,
                      "shard_id": e.shard_id}
                for eid, e in self.kg_edges.items()
            },
        }
        shard_data = {}
        for sid, s in self.shards.items():
            crystal = s.memory.export()
            manifest["shards"].append({
                "id": sid,
                "centroid": s.centroid.tolist(),
                "n_entries": len(s.memory.entries),
            })
            shard_data[sid] = {
                "crystals": crystal.crystals,
                "embeddings": crystal.embeddings,
                "entries": crystal.entries,
                "head_keys": crystal.head_keys,
            }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        _t.save({"manifest": manifest, "shards": shard_data}, path)

    @classmethod
    def from_file(cls, path: str, embedder=None) -> "CrystalForest":
        from .core import CrystalMem, MemoryEntry, ProvenanceStep
        import torch as _t
        bundle = _t.load(path, weights_only=False)
        manifest = bundle["manifest"]
        shard_data = bundle["shards"]

        forest = cls(
            dim=manifest["dim"],
            target_per_crystal=manifest["target_per_crystal"],
            max_per_crystal=manifest["max_per_crystal"],
            n_heads=manifest["n_heads"],
            embedder=embedder,
            bind_algebra=manifest["bind_algebra"],
        )
        for sm in manifest["shards"]:
            sid = sm["id"]
            data = shard_data[sid]
            mem = CrystalMem(
                dim=manifest["dim"], n_heads=manifest["n_heads"],
                bind_algebra=manifest["bind_algebra"], embedder=embedder,
            )
            mem.crystals = [c.detach().clone() for c in data["crystals"]]
            mem.head_keys_by_id = {
                k: [v.detach().clone() for v in vs]
                for k, vs in data["head_keys"].items()
            }
            mem.entries = {}
            mem.entity_facts_idx = {}
            for i, e_dict in enumerate(data["entries"]):
                prov = [ProvenanceStep(**p) for p in e_dict.get("provenance", [])]
                entry = MemoryEntry(
                    id=e_dict["id"], content=e_dict["content"],
                    embedding=data["embeddings"][i].clone(),
                    user_id=e_dict["user_id"], timestamp=e_dict["timestamp"],
                    tags=set(e_dict["tags"]), metadata=e_dict["metadata"],
                    source=e_dict.get("source", "imported"),
                    provenance=prov,
                )
                mem.entries[e_dict["id"]] = entry
                ent = entry.metadata.get("entity")
                if ent:
                    mem.entity_facts_idx.setdefault(ent, set()).add(entry.id)
            shard = Shard(
                shard_id=sid,
                centroid=_t.tensor(sm["centroid"]),
                memory=mem,
            )
            forest.shards[sid] = shard
        for ent, ndata in manifest["kg_nodes"].items():
            forest.kg_nodes[ent] = KGNode(
                entity=ent,
                crystal_centroid=torch.zeros(manifest["dim"]),
                fact_shard_ids=set(ndata["fact_shard_ids"]),
                edges_in=set(ndata["edges_in"]),
                edges_out=set(ndata["edges_out"]),
            )
        for eid, edata in manifest["kg_edges"].items():
            forest.kg_edges[eid] = KGEdge(
                edge_id=eid, src=edata["src"], dst=edata["dst"],
                relation=edata["relation"], shard_id=edata.get("shard_id"),
            )
        return forest
