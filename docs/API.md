# API reference

**Read in:** [English](API.md) · [Русский](API.ru.md)

Every public function, with one-line description and pointer to a working
example. For full signatures, run `help(...)` on the object in Python.

## Core — `crystal_mem`

### `CrystalMem` — main memory class

| Method | Purpose | Example |
|---|---|---|
| `CrystalMem(user_id, dim, n_heads, embedder, ...)` | Construct memory | [01](../examples/01_quickstart.py) |
| `.add(content, tags, metadata, entity, memory_id)` | Write one entry, returns its id | [01](../examples/01_quickstart.py) |
| `.add_many([{content, tags?, ...}, ...])` | Bulk write | — |
| `.add_from_conversation(messages)` | LLM-extract facts (mode="auto" only) | — |
| `.search(query, top_k=5)` | Cosine search over embedding bank, returns `[(entry, score), ...]` | [01](../examples/01_quickstart.py) |
| `.search_by_entity(entity, query, top_k)` | Search restricted to facts about an entity | [04](../examples/04_entities_kg.py) |
| `.recall_via_crystal(memory_id)` | HRR-bottleneck recall (no bank shortcut) | [10](../examples/10_capacity_awareness.py) |
| `.get(memory_id)` | Fetch one entry | [01](../examples/01_quickstart.py) |
| `.get_all()` | All entries as a list | — |
| `.update(memory_id, new_content)` | Forget then re-add same id | — |
| `.forget(memory_id, mode="cryptographic")` | Math-exact removal (Δ ≈ 4×10⁻⁶) | [01](../examples/01_quickstart.py) |
| `.forget_by_filter(predicate)` | Bulk delete matching predicate, returns count | [04](../examples/04_entities_kg.py) |
| `.entity_facts(entity)` | All entries bound to an entity | [04](../examples/04_entities_kg.py) |
| `.entities()` | List entities currently with bound facts | [04](../examples/04_entities_kg.py) |
| `.forget_entity(entity)` | Bulk forget all of an entity's facts | [04](../examples/04_entities_kg.py) |
| `.link(entity_a, entity_b, relation)` | Create a bidirectional KG edge | [04](../examples/04_entities_kg.py) |
| `.snapshot()` | In-memory portable snapshot (`Crystal`) | — |
| `.restore(snapshot)` | Reset state to a snapshot | — |
| `.cap50` (property) | Capacity bound `dim/16` | [10](../examples/10_capacity_awareness.py) |
| `.capacity_used` (property) | Ratio `n_entries / cap50` | [10](../examples/10_capacity_awareness.py) |
| `.recall_estimate()` | empirical recall scaling law | [10](../examples/10_capacity_awareness.py) |
| `.capacity_warning()` | Warning text or `None` | [10](../examples/10_capacity_awareness.py) |
| `.stats()` | Dict of all health metrics | [10](../examples/10_capacity_awareness.py) |

### Portability methods on `CrystalMem`

| Method | Purpose | Example |
|---|---|---|
| `.export(filter=None)` | Returns a `Crystal` snapshot (in-memory) | [05](../examples/05_export_import.py) |
| `.export_file(path, filter=None)` | Write `Crystal` to disk | [05](../examples/05_export_import.py) |
| `CrystalMem.from_file(path, **kwargs)` | Load `Crystal` and construct memory | [05](../examples/05_export_import.py) |
| `CrystalMem.from_crystal(crystal, **kwargs)` | Construct from in-memory `Crystal` | — |
| `.merge(other)` | In-place sum-merge another memory or crystal | [06](../examples/06_federated_merge.py) |
| `.merge_file(path)` | Merge from disk | — |
| `.merge_many([crystals])` | Batch merge | — |
| `.unmerge(other)` | Subtract the contributions of another memory or crystal | [06](../examples/06_federated_merge.py) |
| `.diff(other)` | Returns `(only_in_self, only_in_other, common)` id sets | [05](../examples/05_export_import.py) |

## Embedders — `crystal_mem.embedders`

| Function | Purpose |
|---|---|
| `sentence_transformer_embedder(model_name, target_dim, resize_mode)` | Wrap any sentence-transformers model |
| `qwen3_embedder(target_dim)` | Qwen3-Embedding-0.6B (1024-d Matryoshka, truncatable) |
| `hash_embedder(target_dim)` | Deterministic hash fallback (no semantic match — for tests) |
| `get_default_embedder(target_dim)` | Try real, fall back to hash if no model installed |

## Watermarking — `crystal_mem.watermark`

| Function | Purpose | Example |
|---|---|---|
| `watermark_inject(memory, label, gamma=None, n_carriers=32)` | Stamp a crystal with a label-derived signature | [07](../examples/07_watermark.py) |
| `watermark_detect(memory_or_crystal, label, threshold=4.0)` | Returns `WatermarkResult(detected, z_score, threshold)` | [07](../examples/07_watermark.py) |

## Migration — `crystal_mem.migrate`

| Function | Source format |
|---|---|
| `migrate_from_mem0(memory, source)` | Mem0 export JSON or dict |
| `migrate_from_letta(memory, source, include_recall=False)` | Letta export |
| `migrate_from_zep(memory, source)` | Zep export |
| `migrate_from_chat_jsonl(memory, source_path)` | OpenAI-style `{role, content}` JSONL |

CLI: `python -m crystal_mem.migrate --source <kind> --input <path> --output <crystal_path>`

## Integrations

### `crystal_mem.integrations.langchain`

| Class | Purpose | Example |
|---|---|---|
| `CrystalMessageHistory(memory, session_id, top_k=None)` | `BaseChatMessageHistory` subclass for langchain-core ≥ 1.0 | [08](../examples/08_langchain_basic.py) |
| `CrystalMemoryAdapter(...)` | Legacy alias of `CrystalMessageHistory` | — |

### `crystal_mem.integrations.llamaindex`

| Class | Purpose |
|---|---|
| `CrystalMemBuffer.from_crystal_mem(memory, top_k=5)` | `ChatMemoryBuffer`-compatible wrapper |

### `crystal_mem.mcp_server`

CLI: `python -m crystal_mem.mcp_server --user-id <id> --dim <D> --state <path>`

Exposes 9 MCP tools: `memory_add`, `memory_search`, `memory_get`,
`memory_forget`, `memory_list`, `memory_export`, `memory_merge_file`,
`memory_unmerge_file`, `memory_stats`.

## Forest — `crystal_mem.forest` (experimental)

| Class / method | Purpose |
|---|---|
| `CrystalForest(dim, target_per_crystal, max_per_crystal, n_heads, embedder, ...)` | Bank of crystals + KG overlay |
| `.add(content, entity, tags, metadata)` | Auto-route to best shard |
| `.search(query, top_k, max_shards_to_probe)` | Probe top shards, gather hits |
| `.search_entity(entity, query, top_k)` | KG-routed search |
| `.link(src, dst, relation, context)` | KG edge |
| `.neighbors(entity, depth)` | KG traversal |
| `.compact()` | Force split overfull, merge underfull-similar |
| `.export_file(path)` / `from_file(path)` | Persistence |

See [`CRYSTAL_FOREST.md`](../CRYSTAL_FOREST.md) for the full design + honest verdict.
