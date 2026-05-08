# CrystalForest — Design Document

**Read in:** [English](CRYSTAL_FOREST.md) · [Русский](CRYSTAL_FOREST.ru.md)


> Bank of crystals with auto-placement, compaction, clustering, and KG hybrid.
> Engineering-cool. Honest verdict: **infrastructure, not standalone product.**

## TL;DR

**What it is:** An automatically-managed pool of small CrystalMem instances with
a router that picks where to write/read, compaction that keeps shards healthy,
and a knowledge-graph layer where each entity owns specific shards.

**What problem it solves:** A single CrystalMem at scale (N > 1000) hits the
`cap50 = dim/16` limit per head. HRR-bottleneck retrieval degrades. Either
you accept the loss, increase dim 16× (memory-heavy), or shard.

**What we measured (5 clusters × 2000 entries, dim=384, target=64, max=256):**

| Metric | Flat CrystalMem | CrystalForest |
|---|---|---|
| Ingest 2000 entries | 5.1 s | 17.8 s (3.5× slower) |
| Search latency (mean) | 6.06 ms/query | 17.3 ms/query (4 probes) |
| Cluster purity | n/a | **90%** (min 55%, max 100%) |
| Number of shards | 1 | 13 |
| Auto-compaction | n/a | works (split at max=256) |

**Honest verdict:**
- ✓ Auto-clustering works (90% purity).
- ✓ Auto-split works.
- ✓ KG-routed search works.
- ✗ Latency is **worse**, not better, on this scale.
- ✗ Memory overhead per shard (each holds full state).
- ✓ At N >> cap50, HRR-bottleneck retrieval improves vs flat (each shard stays small).

**Decision:** ship as **experimental infrastructure module**, not product.
Useful when you specifically need: HRR-bottleneck retrieval at scale, KG-graph
hybrid for federated workspaces, or per-cluster privacy isolation.

## Why this idea is real

There's a real gap: a single crystal stops scaling at N > 4·cap50.
Three legitimate solutions exist:

1. **Bigger crystal**: dim=4096 holds cap50=256. But 16× memory.
2. **Multi-head ensemble**: more heads close the gap to softmax recall, at a
   linear compute cost.
3. **Forest**: many small crystals, route to the right one. Fixed per-shard cost.

Approach 3 is what most production vector-DB systems do: HNSW, IVF, ScaNN — all are
"cluster index of small clusters". The forest version brings:

- **Algebraic merge between shards** (federated property holds across the forest too).
- **Per-shard exact forget** without affecting others.
- **Per-shard watermarking** for provenance.
- **KG layer** giving entity-routed search natively.

The math (hierarchical memory routing + KG-style overlay) is well-trodden;
the novelty here is **packaging** as a clean drop-in for CrystalMem.

## Architecture

```
                            CrystalForest
                       ┌───────────────────────┐
                       │  Router (centroid +   │
                       │   capacity scoring)   │
                       └───────────┬───────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
        ┌─────▼────┐         ┌─────▼────┐         ┌─────▼────┐
        │ shard_001│         │ shard_002│   ...   │ shard_NNN│
        │ centroid │         │ centroid │         │ centroid │
        │ memory   │         │ memory   │         │ memory   │
        │ (60/256) │         │ (180/256)│         │ (12/256) │
        └──────────┘         └──────────┘         └──────────┘

                          KG Layer (overlay)
              ┌─────────────────────────────────────┐
              │ alice ──works_at──> acme            │
              │   │                  │              │
              │   └─lives_in─> berlin                │
              │ Each node holds {shard_ids} of facts │
              └─────────────────────────────────────┘
```

### Routing logic

For each new entry:

1. Embed content → emb.
2. Score every existing shard:
   - `centroid_sim(emb, shard.centroid)` — semantic match.
   - `capacity_score = 1 - len(shard) / max_per`.
   - `score = 0.7 * centroid_sim + 0.3 * capacity_score`.
3. If best score < 0.15 AND no under-target shards → spawn a new shard.
4. Else write into the highest-scoring shard.
5. If shard reaches `max_per_crystal` → trigger split.
6. Update centroid as running mean.

### Compaction

Manual `forest.compact()` pass:

- **Split** any shard at max → 2-means on stored embeddings, redistribute.
- **Merge** any two underfull shards (size < target/4) with centroid_sim > 0.5.
- **Re-fit** centroids as means of contents.

### KG Layer

- `add(content, entity="alice")`: shard determined by centroid; entity binds to that shard.
- `link("alice", "acme", relation="works_at")`: KG edge with optional context fact.
- `search_entity("alice", query)`: only probes shards holding alice-bound facts.
- `neighbors("alice", depth=2)`: graph traversal in O(edges).

## What we measured empirically

Setup: 5 cluster templates (Python/Cooking/Travel/Finance/ML), 2000 entries,
dim=384, target=64, max=256.

### Cluster purity — 90% (max 100%, min 55%)

Out of 13 auto-spawned shards, mean shard contains 90% of one cluster type.
This means **auto-routing genuinely separates topics** without supervision.
The 55% minimum is a "mixed" shard at boundary between two semantically-close
clusters (Python and ML overlap).

**This is the strongest positive signal.** Without writing any clustering code,
the forest auto-organizes by content.

### Latency — forest is currently SLOWER

| Method | Mean time/query (50 queries, top_k=5) |
|---|---|
| flat_crystal | **6.06 ms** |
| forest (4 probes / 13 shards) | 17.3 ms |
| forest (probe all shards) | 79.7 ms |

Why slower:
- Each shard has its own embedding cache and cosine loop.
- Even 4 probes × 150 entries each = 600 ops + Python overhead per shard.
- Flat does 1 × 2000 cosine in a single tensor op.

To make forest **win** on latency, we'd need:

1. **Vectorized centroid scoring** — one matmul over all centroids (~13 × dim) → microseconds.
2. **Single-tensor probed search** — concatenate top-K shards into one matrix per query.
3. **Persistent embedding cache** — avoid re-stacking every search.
4. **Lower probe count** — for high-purity clusters, 1-2 probes is enough.

Engineering: 1-2 weeks of work to bring latency parity or 2-3× win.

### Ingest — 3.5× slower

- Flat: 5.1s for 2000 entries (2.5ms/op).
- Forest: 17.8s for 2000 entries (8.9ms/op).

Reason: routing decision + centroid update + occasional shard split.
This is fundamental — fixable with better implementation but won't beat flat.

### Compaction — works correctly

`forest.compact()` returns reasonable stats (split overfull, merge underfull
similar shards). On 2K entries with target=64, 13 shards stayed at 13 because
they were all near max but not over. With smaller target=20 or N=10K+, compact
would do meaningful work.

## Where CrystalForest **wins** (use case map)

| Scenario | Why forest wins |
|---|---|
| **HRR-bottleneck retrieval at N=1K-100K** | Each shard stays in cap50, recall doesn't degrade. Flat fails per the recall law. |
| **Federated workspaces with topic isolation** | Each topic in its own shard → can share / unmerge by topic. |
| **Per-cluster GDPR forget** | Drop entire shard cleanly without scanning others. |
| **KG-hybrid memory** (entity → facts) | Native via `search_entity`. Flat needs extra index. |
| **Crystal marketplace** | Publishers ship topic-shards; subscribers cherry-pick. |
| **Multi-tenant servers** | Each tenant in own shards via `privacy_key` + entity. |

## Where CrystalForest **doesn't win**

| Scenario | Why flat or FAISS is better |
|---|---|
| Simple semantic search at N < 5000 | Flat is faster and simpler. |
| Million-scale RAG | Use FAISS-HNSW or ScaNN. Forest can't compete with HNSW indexing. |
| Latency-critical (sub-1ms) | Forest has Python overhead per shard probe. |
| Single-topic / homogeneous content | Auto-clustering wastes shards. |

## Honest go/no-go

**As a standalone product:** ❌ no.
- The mechanics (clustering + sharding + KG overlay) are well-understood.
- HNSW / Pinecone / Weaviate already solve scale.
- The algebraic properties shine only in specific niches.

**As infrastructure for the product layers in USE_CASES.md:** ⭐⭐ yes, where applicable.
- Cross-Tool Sync: useful for multi-app projection (each app gets its shards).
- Compliance Bus: very useful — per-tenant shards + topic isolation.
- Memory Marketplace: useful — publishers ship topic shards.
- Multi-Agent Workspace: useful — each agent owns shards.

**As a research paper contribution:** maybe.
- "Hypercomplex memory bank with auto-clustering + KG hybrid" is a clean writeup.
- The novelty is mostly packaging — the underlying ideas are well-trodden.

## Recommendation

1. **Keep the prototype** (`forest.py` is working, 8/8 tests pass).
2. **Don't market it separately** — it's plumbing, not product.
3. **Use it WHEN one of these triggers fires:**
   - Customer's data exceeds cap50 by 4×+ → switch single CrystalMem → CrystalForest.
   - Customer needs per-topic export/forget/sharing.
   - Multi-tenant SaaS deployment.
4. **Optimize latency** if it ever gates a real customer:
   - Vectorize centroid scoring.
   - Batched probe search.
   - Better routing thresholds.
5. **Don't over-engineer** clustering algorithms (k-means is fine).

## Code references

- `crystal_mem/forest.py` — implementation.
- `crystal_mem/tests/test_forest.py` — 8 tests covering routing/split/merge/KG/persist.
- `crystal_mem/bench/forest_scale.py` — empirical measurements.

## What this proves about the broader project

CrystalForest demonstrates the gap between **infrastructure** and **product**:
- The library + Forest + KG hybrid is a complete *foundation*.
- Without a focused product layer (Compliance Bus, Memory Marketplace, etc.)
  it's a toolkit looking for a customer.

The product-layer plan in USE_CASES.md is the path forward. Forest is
"Layer 0" — infrastructure that the upper layers can grow into when scale
demands.
