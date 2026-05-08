# Examples

**Read in:** [English](README.md) · [Русский](README.ru.md)

Minimal, runnable, single-purpose snippets. Each file is **20-60 lines** and
demonstrates **one feature**. Copy any file and run it as-is.

For full end-to-end scenarios with multiple features combined, see
[`../crystal_mem/demos/`](../crystal_mem/demos/) instead.

## Run any example

```bash
pip install -e ".[embeddings]"  # core + sentence-transformers

python examples/01_quickstart.py
```

## Index

| File | What it shows |
|---|---|
| [`01_quickstart.py`](01_quickstart.py) | Add facts, search, forget. The 30-second tour. |
| [`02_real_embedder.py`](02_real_embedder.py) | Plug in `sentence-transformers` for real semantic search. |
| [`03_persistence.py`](03_persistence.py) | Auto-save to a `.crystal` file, restore on next run. |
| [`04_entities_kg.py`](04_entities_kg.py) | Per-entity memory + knowledge-graph links. |
| [`05_export_import.py`](05_export_import.py) | Portable crystal files: export, share, import. |
| [`06_federated_merge.py`](06_federated_merge.py) | Combine multiple memories — math-clean sum. |
| [`07_watermark.py`](07_watermark.py) | Provenance signature that survives merge. |
| [`08_langchain_basic.py`](08_langchain_basic.py) | Drop CrystalMem into a LangChain runnable chain. |
| [`09_migrate_from_mem0.py`](09_migrate_from_mem0.py) | Import a Mem0 dump into a fresh CrystalMem. |
| [`10_capacity_awareness.py`](10_capacity_awareness.py) | Read capacity stats and respect the recall law. |

Recommended reading order: top to bottom. Each file builds on what came before.
