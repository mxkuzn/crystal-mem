# CrystalMem

**Язык:** [English](README.md) · [Русский](README.ru.md)

> Research-обоснованная память для LLM-агентов с математически точным
> GDPR-удалением, federated merge и переносимыми crystal-файлами.

[![PyPI](https://img.shields.io/pypi/v/crystal-mem.svg)](https://pypi.org/project/crystal-mem/)
[![Python](https://img.shields.io/pypi/pyversions/crystal-mem.svg)](https://pypi.org/project/crystal-mem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/mxkuzn/crystal-mem/actions/workflows/test.yml/badge.svg)](https://github.com/mxkuzn/crystal-mem/actions/workflows/test.yml)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Быстрые ссылки:**
[Quickstart](docs/QUICKSTART.ru.md) ·
[API](docs/API.ru.md) ·
[Examples](examples/) ·
[Demos](crystal_mem/demos/) ·
[Changelog](CHANGELOG.md) ·
[Use-cases](USE_CASES.ru.md)

```
                       ┌────────────────────┐
   add(fact, entity)──▶│   block-octonion   │──▶ search / search_by_entity
   forget(id)─────────▶│      crystal       │──▶ recall_via_crystal (HRR)
   merge(other)───────▶│  M = Σ bind(k,v)   │──▶ export(*.crystal)
                       └────────────────────┘
                                │
                       ┌────────┴────────┐
                       │ Δ forget = 4e-6 │
                       │ Δ merge  = 9e-8 │
                       │ cap50  = dim/16 │
                       └─────────────────┘
```

## Зачем

LLM-агенты (ChatGPT, Claude, Cursor, Devin, Letta) забывают всё за пределами
context window. Существующие memory-библиотеки (Mem0, MemGPT, Zep, GraphMemory)
теряют recall с ростом N и не умеют **по-настоящему** удалять данные для
GDPR / AI Act.

**CrystalMem** использует block-octonion HRR (Holographic Reduced
Representations). Эмпирические числа — все воспроизводимы из `crystal_mem/bench/`:

| Свойство | Число | Воспроизвести |
|---|---|---|
| Recall vs Mem0-buffer на N=1000 (episodic) | **0.76 vs 0.02 (38× lift)** | `crystal_mem/bench/episodic.py` |
| CoT scratchpad на 50-step цепочках | **100%** | internal test |
| Точное удаление Δ\|M\|/\|M\| | **4×10⁻⁶** | `tests/test_core.py::test_forget_subtracts_from_crystal_exactly` |
| Federated merge идентичен централизованному | Δ = **8.6×10⁻⁸** | `crystal_mem/demos/d3_team_shared_pool.py` |
| Watermark FP-rate | **<0.003%** | `tests/test_watermark.py` |
| Граница ёмкости | cap50 = dim / 16 | `crystal_mem/bench/capacity.py` |
| Закон масштабирования recall (R²) | **0.968** | `crystal_mem/bench/capacity.py` |
| Точность compositional bind | cos > 0.98 | internal test |

## Установка

```bash
pip install crystal-mem                    # ядро
pip install crystal-mem[embeddings]        # + sentence-transformers / Qwen3
pip install crystal-mem[mcp]               # + MCP-сервер для Claude Desktop / Cursor
pip install crystal-mem[langchain]         # + адаптер CrystalMessageHistory
pip install crystal-mem[llamaindex]
pip install crystal-mem[all,dev]           # всё + тестовые тулзы
```

## 30-секундное демо

```python
from crystal_mem import CrystalMem
from crystal_mem.embedders import sentence_transformer_embedder

embed = sentence_transformer_embedder(target_dim=384)
m = CrystalMem(user_id="alice", dim=384, n_heads=4, embedder=embed,
               persist_path="alice.crystal")     # авто-сохранение на каждой мутации

m.add("I prefer Python over Go", entity="alice", tags={"work"})
m.add("Working on Project Atlas — distributed crawler", entity="atlas")
m.add("Allergic to peanuts", entity="alice", tags={"medical"})

for entry, score in m.search("language preferences", top_k=2):
    print(f"  [{score:+.3f}] {entry.content}")

# GDPR-удаление (математически точно, без переиндексации)
m.forget_entity("alice")
```

## Два режима работы

```python
# Режим 1 — explicit (по умолчанию, без LLM-затрат)
m = CrystalMem(mode="explicit")
m.add("user prefers vegan food")

# Режим 2 — auto (извлекает факты через LLM, Mem0-совместимо)
from crystal_mem.extractors import openai_extractor
m = CrystalMem(mode="auto", llm_extractor=openai_extractor("gpt-4o-mini"))
m.add_from_conversation([
    {"role": "user", "content": "I'm vegan, just FYI"},
    {"role": "assistant", "content": "Got it!"},
])
```

## GDPR-grade удаление

```python
m.forget(memory_id)                                       # одна запись
m.forget_entity("customer-c001")                          # всё про сущность
m.forget_by_filter(lambda e: e.user_id == "alice")        # массово
```

Кристалл возвращается к математическому состоянию pre-add (Δ ≈ 4×10⁻⁶).
Это **не** soft-delete и не filter — алгебраически точное вычитание.

## Переносимость памяти — забери AI-память куда угодно

```python
m.export_file("alice.crystal")                # бинарная сериализация

m2 = CrystalMem.from_file("alice.crystal")    # загрузить на другой машине

# Merge (математически идентичен централизованной сборке — Δ ≈ 9×10⁻⁸)
m_combined = CrystalMem(dim=1024, n_heads=4)
m_combined.merge_file("alice_work.crystal")
m_combined.merge_file("alice_personal.crystal")

# Селективный экспорт — только записи с тегом "work"
m.export_file("work.crystal", filter=lambda e: "work" in e.tags)

# Diff между двумя памятями
only_a, only_b, common = m_a.diff(m_b)

# Откат merge'а
m.unmerge_file("merged_in_by_mistake.crystal")
```

## Entity-bound память (knowledge-graph режим)

```python
m.add("uses Postgres", entity="atlas")
m.add("uses Tokio runtime", entity="atlas")
m.add("uses Vue 3", entity="beacon")

m.entity_facts("atlas")                      # все факты про Atlas
m.search_by_entity("atlas", "what database?", top_k=2)

# Двунаправленные связи — KG-style
m.link("alice", "acme", "works_at")
m.link("alice", "berlin", "lives_in")
m.entity_facts("acme")                       # alice работает в acme
```

## MCP-сервер (Claude Desktop / Claude Code / Cursor)

```bash
python -m crystal_mem.mcp_server \
    --user-id alice --dim 1024 --heads 4 \
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

9 инструментов: `memory_add`, `memory_search`, `memory_get`, `memory_forget`,
`memory_list`, `memory_export`, `memory_merge_file`, `memory_unmerge_file`,
`memory_stats`.

## Интеграция с LangChain

```python
from crystal_mem import CrystalMem
from crystal_mem.integrations.langchain import CrystalMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

mem = CrystalMem(user_id="alice", dim=384, embedder=...)
chain_with_memory = RunnableWithMessageHistory(
    chain,
    lambda sid: CrystalMessageHistory(mem, session_id=sid),
    input_messages_key="input", history_messages_key="history",
)
```

Drop-in для `langchain-core >= 1.0`. Изоляция по сессиям + опциональный
retrieval-режим памяти.

## Миграция с Mem0 / Letta / Zep

```python
from crystal_mem.migrate import migrate_from_mem0, migrate_from_letta

m = CrystalMem(user_id="alice", dim=384)
n = migrate_from_mem0(m, "mem0_dump.json")
print(f"Импортировано {n} записей из Mem0")
```

CLI:
```bash
python -m crystal_mem.migrate \
    --source mem0 --input dump.json \
    --output alice.crystal --dim 384
```

Поддерживается: `mem0`, `letta`, `zep`, `chat-jsonl`.

## NEW use case — Memory marketplace

Federated merge + watermarking + селективный фильтр позволяют curated
knowledge-кристаллы как **переносимые артефакты**:

```python
# Сообщество публикует "Python idioms 2026"
publisher = CrystalMem(dim=384, n_heads=4)
for fact in PYTHON_IDIOMS:
    publisher.add(fact, tags={"published"})
watermark_inject(publisher, label="python_idioms_2026")
publisher.export_file("python_idioms.crystal")

# Подписчики делают merge — без переиндексации, без conflict resolution
me = CrystalMem(dim=384, n_heads=4)
me.merge_file("python_idioms.crystal")
me.merge_file("gdpr_compliance_2026.crystal")
me.merge_file("rust_perf_tips.crystal")

# Проверка происхождения
detect = watermark_detect(me, label="python_idioms_2026")
assert detect.detected            # подтверждённый source-of-truth
```

Это **уникально** обеспечивается federated-свойством алгебры — vector-DB-based
решения этого не умеют без перетренировки индекса.

## Watermarking

```python
from crystal_mem.watermark import watermark_inject, watermark_detect

publisher = CrystalMem(...)
watermark_inject(publisher, label="acme_corp_v1")    # γ ≈ √N
publisher.export_file("acme_data.crystal")

# Любой получатель кристалла может верифицировать происхождение
result = watermark_detect("acme_data.crystal", label="acme_corp_v1")
assert result.detected and result.z_score > 4.0
```

## Capacity awareness

```python
print(m.stats())
# {
#   "n_entries": 800, "dim": 1024, "n_heads": 4,
#   "cap50": 64, "capacity_used": 12.5,
#   "recall_estimate": 0.43,           # из эмпирического закона масштабирования
#   ...
# }
print(m.capacity_warning())
# "Capacity exceeded: 800 entries vs cap50=64. Recall будет деградировать
#  по recall_law(N=800, h=4, dim=1024). Увеличь dim или n_heads."
```

## Бенчмарки

Воспроизводимы из репозитория:

```bash
python -m crystal_mem.bench.episodic       # episodic recall — N до 2000
python -m crystal_mem.bench.locomo         # мульти-сессии в стиле LoCoMo
python -m crystal_mem.bench.conversation   # 100-300 turns реалистичного диалога
python -m crystal_mem.bench.capacity       # валидация capacity / scaling-law
python -m crystal_mem.bench.operations     # latency forget / merge / export
python -m crystal_mem.bench.mem0_compat    # CrystalMem vs FAISS vs Mem0-summary
```

Пример вывода (`bench.episodic`):
```
     N method                    recall
   500 mem0_buffer (K=32)         0.10
   500 crystal_h16                1.00      ← 10× lift
  1000 mem0_buffer (K=32)         0.02
  1000 crystal_h16                0.76      ← 38× lift
```

## Демо

```bash
python -m crystal_mem.demos.d1_personal_ai      # авто-сохранение + GDPR-forget
python -m crystal_mem.demos.d2_coding_agent     # память на проект
python -m crystal_mem.demos.d3_team_shared_pool # federated merge
python -m crystal_mem.demos.d4_gdpr_audit       # Article 17 deletion
python -m crystal_mem.demos.d5_portability      # синхронизация между инстансами
python -m crystal_mem.demos.d6_marketplace      # publisher/subscriber кристаллы
```

## Документация

| Где смотреть | Что внутри |
|---|---|
| [`docs/QUICKSTART.ru.md`](docs/QUICKSTART.ru.md) | Старт за 5 минут |
| [`docs/API.ru.md`](docs/API.ru.md) | Каждая публичная функция с описанием в одну строку |
| [`examples/`](examples/) | 10 минимальных сниппетов (20-60 строк каждый) |
| [`crystal_mem/demos/`](demos/) | 6 полных end-to-end сценариев |
| [`USE_CASES.ru.md`](USE_CASES.ru.md) | Продуктовые надстройки поверх библиотеки |
| [`CRYSTAL_FOREST.ru.md`](CRYSTAL_FOREST.ru.md) | Экспериментальный sharded forest (auto-cluster + KG hybrid) |
| [`CHANGELOG.md`](CHANGELOG.md) | Release notes (английский) |
| [`CONTRIBUTING.ru.md`](CONTRIBUTING.ru.md) | Процесс PR / issues |
| [`CODE_OF_CONDUCT.ru.md`](CODE_OF_CONDUCT.ru.md) | Правила сообщества |

## Лицензия

MIT — см. [LICENSE](LICENSE).

## Цитирование

Если CrystalMem помог в работе:

```
@software{crystalmem,
  title  = {CrystalMem: agent memory with exact forget, federated merge, and portable crystal files},
  author = {Maxim Kuznetsov},
  year   = {2026},
  url    = {https://github.com/mxkuzn/crystal-mem}
}
```
