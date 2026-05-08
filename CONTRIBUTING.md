# Contributing to CrystalMem

**Read in:** [English](CONTRIBUTING.md) · [Русский](CONTRIBUTING.ru.md)


Thanks for considering a contribution. This project is empirically-backed memory
infrastructure for LLM agents — small, focused, and tested.

## Quick start

```bash
git clone https://github.com/mxkuzn/crystal-mem.git
cd crystal-mem
pip install -e ".[all,dev]"
pytest
```

All tests should pass in under 30 seconds.

## Reporting issues

When opening an issue:

1. Include version: `python -c "import crystal_mem; print(crystal_mem.__version__)"`
2. Provide minimum reproducible example (≤30 lines).
3. Show actual vs expected behavior with concrete numbers.

## Pull requests

### Scope

- **Bug fixes** — always welcome.
- **Performance improvements** — please include before/after benchmark from `crystal_mem/bench/`.
- **New embedders / extractors / integrations** — keep dependencies optional via pyproject `[extras]`.
- **API changes** — open a discussion issue first. Backwards compatibility matters: this
  library is meant to outlive specific LLM frameworks.

Out of scope:

- Replacing block-octonion algebra. The math is a validated empirical result;
  alternative algebras can be added as `bind_algebra="..."` flags, not by replacement.
- LLM-specific integrations beyond LangChain / LlamaIndex / MCP without justification.

### Workflow

1. Fork + branch (`feat/your-thing` or `fix/issue-NN`).
2. Add tests covering the change (`crystal_mem/tests/`).
3. Run the full suite: `pytest crystal_mem/tests/`.
4. Run benchmarks if performance-sensitive: `python -m crystal_mem.bench.episodic`.
5. Update `CHANGELOG.md` under "Unreleased".
6. Open PR with description of (a) what changes, (b) why, (c) benchmark numbers if applicable.

### Code style

- Type hints on all public APIs.
- Default behavior matches benchmark findings (e.g. `n_heads=4` for the multi-head ensemble path).
- Comments explain *why*, not *what*. Cite the relevant benchmark or test for non-obvious choices.
- No emojis in code. Plain ASCII.

### Tests

- Unit tests preferred over integration tests where possible.
- For LLM/API-dependent paths, use `FakeListLLM` or skip with `pytest.importorskip`.
- A new feature without tests will not be merged.

## Empirical backing

Each feature in CrystalMem has a reproducible benchmark or test. When adding
features, link to the relevant one:

| Feature | Reproduce |
|---|---|
| Recall vs Mem0 buffer (episodic) | `crystal_mem/bench/episodic.py` |
| Multi-head ensemble | `crystal_mem/bench/capacity.py` |
| Exact forget | `tests/test_core.py::test_forget_subtracts_from_crystal_exactly` |
| Federated merge | `crystal_mem/demos/d3_team_shared_pool.py` |
| Watermark | `tests/test_watermark.py` |
| Capacity bound (`cap50 = dim/16`) | `crystal_mem/bench/capacity.py` |
| Compositional / entity-bound | `tests/test_entity_bound.py` |
| `bind_algebra="cl_0_3"` drop-in | covered by core tests |

If you're proposing a feature without empirical backing, that's fine — but mark
it `[experimental]` in docs and disable by default.

## License

By contributing you agree that your contribution will be licensed under the
project's MIT license.
