# Quickstart — 5 минут от нуля до работы

**Язык:** [English](QUICKSTART.md) · [Русский](QUICKSTART.ru.md)

## Установка

```bash
# Ядро
pip install crystal-mem

# С реальными семантическими эмбеддингами (рекомендуется)
pip install crystal-mem[embeddings]

# Со всем (MCP-сервер, LangChain, LlamaIndex, экстракторы)
pip install crystal-mem[all]
```

## 30-секундный тур

```python
from crystal_mem import CrystalMem
from crystal_mem.embedders import sentence_transformer_embedder

m = CrystalMem(
    user_id="alice",
    dim=384,
    n_heads=4,
    embedder=sentence_transformer_embedder(target_dim=384),
)

m.add("I prefer Python over Go for backend services", tags={"work"})
m.add("Allergic to peanuts", tags={"medical"})
m.add("Working on Project Atlas — distributed crawler", entity="atlas")

# Семантический поиск
for entry, score in m.search("what languages do I like?", top_k=2):
    print(f"  [{score:+.3f}] {entry.content}")

# GDPR-точное удаление (математически точно, без переиндексации)
m.forget_by_filter(lambda e: "medical" in e.tags)
```

## Типовые задачи

### Сохранение между запусками

```python
m = CrystalMem(
    user_id="alice", dim=384, n_heads=4,
    persist_path="/path/to/alice.crystal",   # авто-сохранение на каждой мутации
)
```

### Per-entity scoping (knowledge-graph режим)

```python
m.add("uses Postgres", entity="atlas")
m.add("uses Tokio runtime", entity="atlas")
m.add("uses Vue 3", entity="beacon")

m.entity_facts("atlas")                                 # все факты про atlas
m.search_by_entity("atlas", "what database?", top_k=2)  # поиск в рамках entity

m.link("alice", "acme", "works_at")                     # KG-связи
m.link("bob",   "acme", "works_at")
m.entity_facts("acme")                                  # обе alice и bob связаны
```

### Переносимость памяти

```python
m.export_file("alice.crystal")                          # сериализация на диск

m2 = CrystalMem.from_file("alice.crystal")              # загрузить на другой машине

m2.merge_file("partner.crystal")                        # federated объединение
m2.unmerge_file("merged_by_mistake.crystal")            # откатить merge

# Селективный экспорт — только записи с тегом "work"
m.export_file("work.crystal", filter=lambda e: "work" in e.tags)
```

### Подключение к LangChain

```python
from crystal_mem.integrations.langchain import CrystalMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

mem = CrystalMem(dim=384, n_heads=4)
chain_with_memory = RunnableWithMessageHistory(
    your_chain,
    lambda sid: CrystalMessageHistory(mem, session_id=sid),
    input_messages_key="input",
    history_messages_key="history",
)
```

### Миграция с другого memory-сервиса

```python
from crystal_mem.migrate import migrate_from_mem0

m = CrystalMem(user_id="alice", dim=384)
n = migrate_from_mem0(m, "mem0_dump.json")
print(f"Импортировано {n} записей")
```

CLI-эквивалент:

```bash
python -m crystal_mem.migrate \
    --source mem0 --input dump.json \
    --output alice.crystal --dim 384
```

### Запуск MCP-сервера (Claude Desktop / Cursor / etc.)

```bash
python -m crystal_mem.mcp_server \
    --user-id alice --dim 1024 \
    --state ~/.crystal_mem/alice.crystal
```

В config Claude Desktop:

```json
{
  "mcpServers": {
    "crystal-mem": {
      "command": "python",
      "args": ["-m", "crystal_mem.mcp_server",
               "--user-id", "alice",
               "--state", "/Users/alice/.crystal_mem/alice.crystal"]
    }
  }
}
```

Доступные tools: `memory_add`, `memory_search`, `memory_get`, `memory_forget`,
`memory_list`, `memory_export`, `memory_merge_file`, `memory_unmerge_file`,
`memory_stats`.

## Дальше

- [API reference](API.ru.md) — каждая публичная функция в одну строку.
- [`../examples/`](../examples/) — 10 single-purpose сниппетов.
- [`../crystal_mem/demos/`](../crystal_mem/demos/) — 6 полных сценариев, включая
  [memory marketplace](../crystal_mem/demos/d6_marketplace.py).
- [Гайд по ёмкости](../examples/10_capacity_awareness.py) — когда увеличивать
  dim vs heads vs forest.
