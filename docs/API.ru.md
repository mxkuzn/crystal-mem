# API reference

**Язык:** [English](API.md) · [Русский](API.ru.md)

Каждая публичная функция, описание в одну строку, ссылка на работающий
пример. Полные сигнатуры через `help(...)` в Python.

## Ядро — `crystal_mem`

### `CrystalMem` — основной класс памяти

| Метод | Что делает | Пример |
|---|---|---|
| `CrystalMem(user_id, dim, n_heads, embedder, ...)` | Конструктор | [01](../examples/01_quickstart.py) |
| `.add(content, tags, metadata, entity, memory_id)` | Записать запись, возвращает её id | [01](../examples/01_quickstart.py) |
| `.add_many([{content, tags?, ...}, ...])` | Bulk запись | — |
| `.add_from_conversation(messages)` | LLM-извлечение фактов (только mode="auto") | — |
| `.search(query, top_k=5)` | Cosine-поиск по бэнку эмбеддингов, `[(entry, score), ...]` | [01](../examples/01_quickstart.py) |
| `.search_by_entity(entity, query, top_k)` | Поиск только среди фактов про entity | [04](../examples/04_entities_kg.py) |
| `.recall_via_crystal(memory_id)` | HRR-bottleneck recall (без bank-shortcut) | [10](../examples/10_capacity_awareness.py) |
| `.get(memory_id)` | Получить одну запись | [01](../examples/01_quickstart.py) |
| `.get_all()` | Все записи списком | — |
| `.update(memory_id, new_content)` | Forget + re-add с тем же id | — |
| `.forget(memory_id, mode="cryptographic")` | Math-точное удаление (Δ ≈ 4×10⁻⁶) | [01](../examples/01_quickstart.py) |
| `.forget_by_filter(predicate)` | Bulk-удаление по предикату, возвращает count | [04](../examples/04_entities_kg.py) |
| `.entity_facts(entity)` | Все записи привязанные к entity | [04](../examples/04_entities_kg.py) |
| `.entities()` | Список entities с привязанными фактами | [04](../examples/04_entities_kg.py) |
| `.forget_entity(entity)` | Bulk-удаление всех фактов про entity | [04](../examples/04_entities_kg.py) |
| `.link(entity_a, entity_b, relation)` | Создать двунаправленный KG-edge | [04](../examples/04_entities_kg.py) |
| `.snapshot()` | In-memory снапшот (`Crystal`) | — |
| `.restore(snapshot)` | Восстановить состояние из снапшота | — |
| `.cap50` (property) | Граница ёмкости `dim/16` | [10](../examples/10_capacity_awareness.py) |
| `.capacity_used` (property) | Доля `n_entries / cap50` | [10](../examples/10_capacity_awareness.py) |
| `.recall_estimate()` | Эмпирический закон масштабирования recall | [10](../examples/10_capacity_awareness.py) |
| `.capacity_warning()` | Warning text или `None` | [10](../examples/10_capacity_awareness.py) |
| `.stats()` | Dict со всеми health-метриками | [10](../examples/10_capacity_awareness.py) |

### Методы переносимости на `CrystalMem`

| Метод | Что делает | Пример |
|---|---|---|
| `.export(filter=None)` | Возвращает `Crystal` (in-memory) | [05](../examples/05_export_import.py) |
| `.export_file(path, filter=None)` | Записать `Crystal` на диск | [05](../examples/05_export_import.py) |
| `CrystalMem.from_file(path, **kwargs)` | Загрузить `Crystal` и сконструировать память | [05](../examples/05_export_import.py) |
| `CrystalMem.from_crystal(crystal, **kwargs)` | Сконструировать из in-memory `Crystal` | — |
| `.merge(other)` | In-place sum-merge другой памяти/кристалла | [06](../examples/06_federated_merge.py) |
| `.merge_file(path)` | Merge с диска | — |
| `.merge_many([crystals])` | Batch-merge | — |
| `.unmerge(other)` | Вычесть вклад другой памяти/кристалла | [06](../examples/06_federated_merge.py) |
| `.diff(other)` | Возвращает `(only_in_self, only_in_other, common)` сеты id | [05](../examples/05_export_import.py) |

## Эмбеддеры — `crystal_mem.embedders`

| Функция | Что делает |
|---|---|
| `sentence_transformer_embedder(model_name, target_dim, resize_mode)` | Wrap любой sentence-transformers модели |
| `qwen3_embedder(target_dim)` | Qwen3-Embedding-0.6B (1024-d Matryoshka, обрезаемый) |
| `hash_embedder(target_dim)` | Детерминированный hash (без semantic match — для тестов) |
| `get_default_embedder(target_dim)` | Попробовать real, fallback на hash если нет модели |

## Watermarking — `crystal_mem.watermark`

| Функция | Что делает | Пример |
|---|---|---|
| `watermark_inject(memory, label, gamma=None, n_carriers=32)` | Поставить лейбл-производную сигнатуру | [07](../examples/07_watermark.py) |
| `watermark_detect(memory_or_crystal, label, threshold=4.0)` | Возвращает `WatermarkResult(detected, z_score, threshold)` | [07](../examples/07_watermark.py) |

## Миграция — `crystal_mem.migrate`

| Функция | Source-формат |
|---|---|
| `migrate_from_mem0(memory, source)` | Mem0 export JSON или dict |
| `migrate_from_letta(memory, source, include_recall=False)` | Letta export |
| `migrate_from_zep(memory, source)` | Zep export |
| `migrate_from_chat_jsonl(memory, source_path)` | OpenAI-style `{role, content}` JSONL |

CLI: `python -m crystal_mem.migrate --source <kind> --input <path> --output <crystal_path>`

## Интеграции

### `crystal_mem.integrations.langchain`

| Класс | Что делает | Пример |
|---|---|---|
| `CrystalMessageHistory(memory, session_id, top_k=None)` | Подкласс `BaseChatMessageHistory` для langchain-core ≥ 1.0 | [08](../examples/08_langchain_basic.py) |
| `CrystalMemoryAdapter(...)` | Legacy alias для `CrystalMessageHistory` | — |

### `crystal_mem.integrations.llamaindex`

| Класс | Что делает |
|---|---|
| `CrystalMemBuffer.from_crystal_mem(memory, top_k=5)` | `ChatMemoryBuffer`-совместимый wrapper |

### `crystal_mem.mcp_server`

CLI: `python -m crystal_mem.mcp_server --user-id <id> --dim <D> --state <path>`

Экспортирует 9 MCP-tools: `memory_add`, `memory_search`, `memory_get`,
`memory_forget`, `memory_list`, `memory_export`, `memory_merge_file`,
`memory_unmerge_file`, `memory_stats`.

## Forest — `crystal_mem.forest` (experimental)

| Класс / метод | Что делает |
|---|---|
| `CrystalForest(dim, target_per_crystal, max_per_crystal, n_heads, embedder, ...)` | Bank кристаллов + KG-overlay |
| `.add(content, entity, tags, metadata)` | Авто-routing в лучший shard |
| `.search(query, top_k, max_shards_to_probe)` | Probe top-K shards, сбор хитов |
| `.search_entity(entity, query, top_k)` | KG-routed поиск |
| `.link(src, dst, relation, context)` | KG edge |
| `.neighbors(entity, depth)` | KG обход |
| `.compact()` | Force split overfull, merge underfull-similar |
| `.export_file(path)` / `from_file(path)` | Сериализация |

См. [`CRYSTAL_FOREST.ru.md`](../CRYSTAL_FOREST.ru.md) для полного дизайна + честного verdict.
