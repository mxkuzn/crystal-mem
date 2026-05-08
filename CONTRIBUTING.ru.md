# Контрибьютинг в CrystalMem

**Язык:** [English](CONTRIBUTING.md) · [Русский](CONTRIBUTING.ru.md)

Спасибо за интерес. Проект — эмпирически-обоснованная memory-инфраструктура
для LLM-агентов. Маленькая, сфокусированная, протестированная.

## Быстрый старт

```bash
git clone https://github.com/mxkuzn/crystal-mem.git
cd crystal-mem
pip install -e ".[all,dev]"
pytest
```

Все тесты должны пройти за полминуты.

## Сообщения об ошибках

Когда открываешь issue:

1. Укажи версию: `python -c "import crystal_mem; print(crystal_mem.__version__)"`
2. Минимальный воспроизводимый пример (≤30 строк).
3. Покажи реальное vs ожидаемое поведение с конкретными числами.

## Pull requests

### Что в скоупе

- **Bug fixes** — всегда welcome.
- **Performance** — приложи before/after из `crystal_mem/bench/`.
- **Новые embedders / extractors / интеграции** — держи зависимости опциональными
  через pyproject `[extras]`.
- **API changes** — сначала открой discussion-issue. Backwards-совместимость
  важна: библиотека должна пережить конкретные LLM-фреймворки.

Не в скоупе:

- Замена block-octonion алгебры. Математика — валидированный эмпирический
  результат; альтернативные алгебры могут быть добавлены как флаги
  `bind_algebra="..."`, но не заменять текущую.
- LLM-специфичные интеграции вне LangChain / LlamaIndex / MCP без
  обоснования.

### Workflow

1. Fork + branch (`feat/your-thing` или `fix/issue-NN`).
2. Добавь тесты под изменение (`crystal_mem/tests/`).
3. Прогон полного suite: `pytest crystal_mem/tests/`.
4. Запусти бенчмарки если performance-чувствительно: `python -m crystal_mem.bench.episodic`.
5. Обнови `CHANGELOG.md` под "Unreleased".
6. Открой PR с описанием (a) что меняется, (b) почему, (c) числа бенчмарков
   если применимо.

### Стиль кода

- Type hints на всех публичных API.
- Default-поведение соответствует benchmark-findings (например, `n_heads=4`
  для multi-head ensemble path).
- Комментарии объясняют *почему*, не *что*. Цитируй соответствующий бенчмарк
  или тест для неочевидных решений.
- Никаких emoji в коде. Plain ASCII.

### Тесты

- Unit-тесты предпочтительнее integration где можно.
- Для LLM/API-зависимых путей используй `FakeListLLM` или `pytest.importorskip`.
- Новая фича без тестов **не** будет смержена.

## Эмпирическое обоснование

Каждая фича CrystalMem имеет воспроизводимый бенчмарк или тест. При добавлении
новой фичи привязывай её к соответствующему бенчмарку:

| Фича | Воспроизвести |
|---|---|
| Recall vs Mem0 buffer (episodic) | `crystal_mem/bench/episodic.py` |
| Multi-head ensemble | `crystal_mem/bench/capacity.py` |
| Точное удаление | `tests/test_core.py::test_forget_subtracts_from_crystal_exactly` |
| Federated merge | `crystal_mem/demos/d3_team_shared_pool.py` |
| Watermark | `tests/test_watermark.py` |
| Граница ёмкости (`cap50 = dim/16`) | `crystal_mem/bench/capacity.py` |
| Compositional / entity-bound | `tests/test_entity_bound.py` |
| `bind_algebra="cl_0_3"` drop-in | покрыт core-тестами |

Если предлагаешь фичу без эмпирического обоснования — это окей, но помечай
как `[experimental]` в доке и отключай по умолчанию.

## Лицензия

Контрибьютя, ты соглашаешься что твой вклад будет под MIT лицензией проекта.
