# Documentation

**Read in:** [English](README.md) · [Русский](README.ru.md)

Tour of the docs in recommended reading order.

| If you want to... | Read |
|---|---|
| Get started in 5 minutes | [QUICKSTART.md](QUICKSTART.md) |
| Understand the public API | [API.md](API.md) |
| See what each method does | [`../examples/`](../examples/) — 10 minimal scripts |
| See full scenarios | [`../crystal_mem/demos/`](../crystal_mem/demos/) — 6 end-to-end demos |
| Understand product layers | [`../USE_CASES.md`](../USE_CASES.md) |
| Use the experimental forest | [`../CRYSTAL_FOREST.md`](../CRYSTAL_FOREST.md) |
| Contribute | [`../CONTRIBUTING.md`](../CONTRIBUTING.md) |
| Browse the changelog | [`../CHANGELOG.md`](../CHANGELOG.md) |
| Migrate from Mem0/Letta/Zep | [examples/09_migrate_from_mem0.py](../examples/09_migrate_from_mem0.py) + `python -m crystal_mem.migrate --help` |

## Empirical backing

Every claim in the README and this documentation has a reproducible benchmark
or test:

| Claim | Reproduce |
|---|---|
| Recall vs Mem0 at N=1000 — 38× lift | `crystal_mem/bench/episodic.py` |
| 100% on multi-step CoT scratchpad | `tests/test_core.py` (multi-head recall) |
| Exact forget Δ ≈ 4×10⁻⁶ | `tests/test_core.py::test_forget_subtracts_from_crystal_exactly` |
| Federated merge Δ = 8.6×10⁻⁸ | `crystal_mem/demos/d3_team_shared_pool.py` |
| Watermark FP <0.003% | `tests/test_watermark.py` |
| Capacity bound `cap50 = dim/16` | `crystal_mem/bench/capacity.py` |
| Recall scaling law `1 − exp(−α·h·dim/N)` | `crystal_mem/bench/capacity.py` |
| Compositional bind cos > 0.98 | `tests/test_entity_bound.py` |
| `bind_algebra="cl_0_3"` drop-in | core tests |
