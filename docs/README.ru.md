# Документация

**Язык:** [English](README.md) · [Русский](README.ru.md)

Обзор документов в рекомендуемом порядке чтения.

| Если хочешь... | Читай |
|---|---|
| Стартовать за 5 минут | [QUICKSTART.ru.md](QUICKSTART.ru.md) |
| Понять публичный API | [API.ru.md](API.ru.md) |
| Увидеть что делает каждый метод | [`../examples/`](../examples/) — 10 минимальных скриптов |
| Полные сценарии | [`../crystal_mem/demos/`](../crystal_mem/demos/) — 6 end-to-end демо |
| Понять продуктовые надстройки | [`../USE_CASES.ru.md`](../USE_CASES.ru.md) |
| Использовать experimental forest | [`../CRYSTAL_FOREST.ru.md`](../CRYSTAL_FOREST.ru.md) |
| Контрибьютить | [`../CONTRIBUTING.ru.md`](../CONTRIBUTING.ru.md) |
| Посмотреть changelog | [`../CHANGELOG.md`](../CHANGELOG.md) |
| Мигрировать с Mem0/Letta/Zep | [examples/09_migrate_from_mem0.py](../examples/09_migrate_from_mem0.py) + `python -m crystal_mem.migrate --help` |

## Эмпирическое обоснование

Каждое утверждение в README и документации имеет воспроизводимый бенчмарк
или тест:

| Утверждение | Воспроизвести |
|---|---|
| Recall vs Mem0 на N=1000 — 38× lift | `crystal_mem/bench/episodic.py` |
| 100% на multi-step CoT scratchpad | `tests/test_core.py` (multi-head recall) |
| Точное удаление Δ ≈ 4×10⁻⁶ | `tests/test_core.py::test_forget_subtracts_from_crystal_exactly` |
| Federated merge Δ = 8.6×10⁻⁸ | `crystal_mem/demos/d3_team_shared_pool.py` |
| Watermark FP <0.003% | `tests/test_watermark.py` |
| Граница ёмкости `cap50 = dim/16` | `crystal_mem/bench/capacity.py` |
| Закон масштабирования recall `1 − exp(−α·h·dim/N)` | `crystal_mem/bench/capacity.py` |
| Compositional bind cos > 0.98 | `tests/test_entity_bound.py` |
| `bind_algebra="cl_0_3"` drop-in | core-тесты |
