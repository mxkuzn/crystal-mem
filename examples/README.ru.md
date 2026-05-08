# Examples

**Язык:** [English](README.md) · [Русский](README.ru.md)

Минимальные, запускаемые сниппеты с одной целью каждый. Каждый файл —
**20-60 строк**, демонстрирует **одну фичу**. Копируй и запускай.

Для полных end-to-end сценариев со множеством фич смотри
[`../crystal_mem/demos/`](../crystal_mem/demos/) вместо этого.

## Запуск любого примера

```bash
pip install -e ".[embeddings]"  # ядро + sentence-transformers

python examples/01_quickstart.py
```

## Индекс

| Файл | Что показывает |
|---|---|
| [`01_quickstart.py`](01_quickstart.py) | Добавить факты, поиск, забыть. 30-секундный тур. |
| [`02_real_embedder.py`](02_real_embedder.py) | Подключить `sentence-transformers` для real semantic search. |
| [`03_persistence.py`](03_persistence.py) | Авто-сохранение в `.crystal` файл, восстановление при следующем запуске. |
| [`04_entities_kg.py`](04_entities_kg.py) | Per-entity память + knowledge-graph связи. |
| [`05_export_import.py`](05_export_import.py) | Переносимые crystal-файлы: экспорт, шаринг, импорт. |
| [`06_federated_merge.py`](06_federated_merge.py) | Объединение нескольких памятей — math-clean sum. |
| [`07_watermark.py`](07_watermark.py) | Provenance-сигнатура, переживающая merge. |
| [`08_langchain_basic.py`](08_langchain_basic.py) | Подключение CrystalMem в LangChain runnable chain. |
| [`09_migrate_from_mem0.py`](09_migrate_from_mem0.py) | Импорт Mem0-дампа в свежий CrystalMem. |
| [`10_capacity_awareness.py`](10_capacity_awareness.py) | Чтение статистики ёмкости и закон recall. |

Рекомендуемый порядок чтения: сверху вниз. Каждый файл строится на предыдущих.
