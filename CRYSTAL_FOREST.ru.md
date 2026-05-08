# CrystalForest — Design Document

**Язык:** [English](CRYSTAL_FOREST.md) · [Русский](CRYSTAL_FOREST.ru.md)

> Bank кристаллов с auto-placement, compaction, кластеризацией и KG-hybrid.
> Инженерно круто. Честный verdict: **infrastructure, не standalone product.**

## TL;DR

**Что это:** автоматически управляемый pool маленьких CrystalMem-инстансов с
роутером, выбирающим куда писать/читать; compaction поддерживающий shards
здоровыми; KG-слой где каждая entity владеет своими shards.

**Какую проблему решает:** одиночный CrystalMem на масштабе (N > 1000) упирается
в `cap50 = dim/16` per head. HRR-bottleneck retrieval деградирует. Либо ты
принимаешь потерю, либо увеличиваешь dim в 16 раз (memory-heavy), либо шардишь.

**Что мы измерили (5 кластеров × 2000 записей, dim=384, target=64, max=256):**

| Метрика | Flat CrystalMem | CrystalForest |
|---|---|---|
| Ingest 2000 записей | 5.1s | 17.8s (3.5× медленнее) |
| Search latency (mean) | 6.06ms/query | 17.3ms/query (4 probes) |
| Cluster purity | n/a | **90%** (min 55%, max 100%) |
| Number of shards | 1 | 13 |
| Auto-compaction | n/a | работает (split at max=256) |

**Честный verdict:**
- ✓ Auto-clustering работает (90% purity).
- ✓ Auto-split работает.
- ✓ KG-routed search работает.
- ✗ Latency **хуже**, не лучше, на этом масштабе.
- ✗ Memory overhead per shard (каждый держит full state).
- ✓ На N >> cap50 HRR-bottleneck retrieval улучшается vs flat (каждый shard остаётся маленьким).

**Решение:** ship как **experimental infrastructure module**, не product.
Полезен когда специально нужно: HRR-bottleneck retrieval at scale, KG-graph
hybrid для federated workspaces, или per-cluster privacy-изоляция.

## Почему идея реальна

Здесь есть реальный gap: один кристалл перестаёт скейлиться на N > 4·cap50.
Три легитимных решения существуют:

1. **Bigger crystal**: dim=4096 держит cap50=256. Но 16× memory.
2. **Multi-head ensemble**: больше голов закрывают gap до softmax recall, при
   линейной compute-стоимости.
3. **Forest**: много маленьких кристаллов, route в правильный. Фиксированная per-shard стоимость.

Approach 3 — то что делают большинство production vector-DB систем: HNSW, IVF,
ScaNN — все есть "cluster index of small clusters". Forest-версия даёт:

- **Algebraic merge между shards** (federated-свойство держится через forest).
- **Per-shard exact forget** не affecting других.
- **Per-shard watermark** для provenance.
- **KG-слой** дающий entity-routed search natively.

Математика (hierarchical memory routing + KG-overlay) хорошо изученная;
новизна — **упаковка** как clean drop-in для CrystalMem.

## Архитектура

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

                          KG-Layer (overlay)
              ┌─────────────────────────────────────┐
              │ alice ──works_at──> acme            │
              │   │                  │              │
              │   └─lives_in─> berlin                │
              │ Каждая нода держит {shard_ids}      │
              └─────────────────────────────────────┘
```

### Routing-логика

Для каждой новой записи:

1. Embed content → emb.
2. Score каждого существующего shard:
   - `centroid_sim(emb, shard.centroid)` — semantic match.
   - `capacity_score = 1 - len(shard) / max_per`.
   - `score = 0.7 * centroid_sim + 0.3 * capacity_score`.
3. Если best score < 0.15 И нет under-target shards → spawn новый shard.
4. Иначе пишем в highest-scoring shard.
5. Если shard достиг `max_per_crystal` → trigger split.
6. Update centroid как running mean.

### Compaction

Manual `forest.compact()`:

- **Split** любой shard at max → 2-means на хранимых эмбеддингах, redistribute.
- **Merge** любые два underfull shards (size < target/4) с centroid_sim > 0.5.
- **Re-fit** centroids как mean'ы содержимого.

### KG-слой

- `add(content, entity="alice")`: shard выбирается по centroid; entity binds к нему.
- `link("alice", "acme", relation="works_at")`: KG-edge с опциональным context fact.
- `search_entity("alice", query)`: probe'ит только shards где alice-bound facts.
- `neighbors("alice", depth=2)`: graph traversal в O(edges).

## Что мы измерили эмпирически

Setup: 5 кластерных шаблонов (Python/Cooking/Travel/Finance/ML), 2000 записей,
dim=384, target=64, max=256.

### Cluster purity — 90% (max 100%, min 55%)

Из 13 авто-spawned shards, средний shard содержит 90% одного типа кластера.
Это значит **auto-routing реально разделяет темы** без supervision. 55% min —
это "mixed" shard на границе между двумя семантически близкими кластерами
(Python и ML overlap'ятся).

**Это самый сильный позитивный сигнал.** Без написания clustering-кода forest
автоматически организует по контенту.

### Latency — forest сейчас МЕДЛЕННЕЕ

| Method | Mean time/query (50 queries, top_k=5) |
|---|---|
| flat_crystal | **6.06 ms** |
| forest (4 probes / 13 shards) | 17.3 ms |
| forest (probe all shards) | 79.7 ms |

Почему медленнее:
- Каждый shard со своим embedding-кэшем и cosine-loop'ом.
- Даже 4 probes × 150 entries each = 600 ops + Python overhead per shard.
- Flat делает 1 × 2000 cosine в одной tensor-операции.

Чтобы forest **выиграл** в latency, нужно:

1. **Векторизованный centroid scoring** — один matmul по всем centroid'ам (~13 × dim) → microseconds.
2. **Single-tensor probed search** — concatenate top-K shards в одну матрицу per query.
3. **Persistent embedding cache** — избегать re-stack'инга каждый search.
4. **Меньше probe count** — для high-purity кластеров 1-2 probes достаточно.

Engineering: 1-2 недели работы для latency parity или 2-3× win.

### Ingest — 3.5× медленнее

- Flat: 5.1s for 2000 entries (2.5ms/op).
- Forest: 17.8s for 2000 entries (8.9ms/op).

Причина: routing-decision + centroid update + изредка shard-split. Это
fundamental — fixable with better impl но не побьёт flat.

### Compaction — работает корректно

`forest.compact()` возвращает разумную статистику (split overfull, merge
underfull-similar shards). На 2K entries с target=64 13 shards остались 13
потому что все были near max но not over. С smaller target=20 или N=10K+
compact сделал бы значимую работу.

## Где CrystalForest **выигрывает** (use case map)

| Сценарий | Почему forest выигрывает |
|---|---|
| **HRR-bottleneck retrieval at N=1K-100K** | Каждый shard остаётся в cap50, recall не деградирует. Flat падает по recall-закону. |
| **Federated workspaces с topic-изоляцией** | Каждая тема в своём shard → можно шарить / unmerge by topic. |
| **Per-cluster GDPR forget** | Drop entire shard cleanly без сканирования других. |
| **KG-hybrid memory** (entity → facts) | Native через `search_entity`. Flat нужен extra index. |
| **Crystal marketplace** | Publishers ship topic-shards; subscribers cherry-pick. |
| **Multi-tenant servers** | Каждый tenant в своих shards через `privacy_key` + entity. |

## Где CrystalForest **не выигрывает**

| Сценарий | Почему flat или FAISS лучше |
|---|---|
| Простой semantic search at N < 5000 | Flat быстрее и проще. |
| Million-scale RAG | Используй FAISS-HNSW или ScaNN. Forest не конкурирует с HNSW indexing. |
| Latency-critical (sub-1ms) | Forest имеет Python overhead per shard probe. |
| Single-topic / homogeneous content | Auto-clustering впустую расходует shards. |

## Честный go/no-go

**Как standalone продукт:** ❌ нет.
- Механика (clustering + sharding + KG overlay) хорошо изучена.
- HNSW / Pinecone / Weaviate уже решают scale.
- Алгебраические свойства светят только в специфичных нишах.

**Как infrastructure для продуктовых layers в USE_CASES.ru.md:** ⭐⭐ да, где применимо.
- Cross-Tool Sync: полезно для multi-app projection (каждое app свои shards).
- Compliance Bus: очень полезно — per-tenant shards + topic-изоляция.
- Memory Marketplace: полезно — publishers ship topic shards.
- Multi-Agent Workspace: полезно — каждый агент свои shards.

**Как research paper contribution:** возможно.
- "Hypercomplex memory bank with auto-clustering + KG hybrid" — clean writeup.
- Новизна в основном в упаковке — базовые идеи хорошо изучены.

## Рекомендация

1. **Сохранить прототип** (`forest.py` рабочий, 8/8 тестов pass).
2. **Не маркетить отдельно** — это plumbing, не продукт.
3. **Использовать КОГДА** один из триггеров:
   - Customer'ские данные превышают cap50 в 4×+ → switch single CrystalMem → CrystalForest.
   - Customer нужен per-topic export/forget/sharing.
   - Multi-tenant SaaS deployment.
4. **Оптимизировать latency** только когда gates реального customer'а:
   - Vectorize centroid scoring.
   - Batched probe search.
   - Better routing thresholds.
5. **Не over-engineer** clustering algorithms (k-means нормально).

## Code references

- `crystal_mem/forest.py` — реализация.
- `crystal_mem/tests/test_forest.py` — 8 тестов покрывающих routing/split/merge/KG/persist.
- `crystal_mem/bench/forest_scale.py` — эмпирические измерения.

## Что это доказывает про проект в целом

CrystalForest демонстрирует gap между **инфраструктурой** и **продуктом**:
- Библиотека + Forest + KG-hybrid — полный *foundation*.
- Без сфокусированного продуктового слоя (Compliance Bus, Memory Marketplace,
  etc.) это toolkit ищущий customer'а.

Layer-план из USE_CASES.ru.md — путь вперёд. Forest — это "Layer 0" —
infrastructure которая может расти в upper layers когда scale потребует.
