# Changelog

All notable changes to CrystalMem will be documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project uses [SemVer](https://semver.org/).

## [Unreleased]

### Added
- LoCoMo-style multi-session conversation benchmark.
- Mem0 / Letta migration scripts.
- LangChain `RunnableWithMessageHistory` end-to-end test using `FakeListLLM`.
- CI workflow (GitHub Actions) running pytest on push/PR.

## [0.1.0] - 2026-05-08

### Added — initial release

- Core `CrystalMem` class with `add` / `search` / `forget` / `update` / `get_all`.
- Two operating modes: `mode="explicit"` and `mode="auto"` (LLM-extraction).
- Multi-head HRR ensemble (`n_heads`) — closes the gap to softmax recall.
- Three bind algebras: `octonion` (default), `cl_0_3`, `duality`.
- Privacy hash via `privacy_key=` — without the key, recall ≈ 0.
- Exact algebraic forget — math-clean Δ ≈ 4×10⁻⁶.
- Capacity awareness: `cap50 = dim/16`, `recall_estimate()` from empirical law.
- **Memory portability**: `Crystal` file format with magic header + json metadata
  + torch.save payload. `export()` / `from_file()` / `merge()` / `unmerge()` /
  `diff()` / filtered export.
- **Federated merge** mathematically identical to centralized build
  (empirical Δ ≈ 9×10⁻⁸).
- **Provenance lineage** — each `MemoryEntry` carries a `provenance` chain
  describing add / merge / import history.
- **Watermarking** module — z-score injection + detect, FP < 0.003%,
  survives export and merge.
- **Entity-bound** API: `m.add(content, entity=...)`, `m.entity_facts()`,
  `m.search_by_entity()`, `m.forget_entity()`, `m.link(a, b, rel)` (KG-style).
- **Snapshots** (`m.snapshot()` / `m.restore()`) — time-travel via crystal commutativity.
- **HRR-bottleneck recall** (`recall_via_crystal`) — extract via unbind+cleanup
  without bank shortcut. Validates capacity at runtime.
- Real embedder integration: sentence-transformers default + Qwen3 option +
  hash fallback. Auto-resize via Matryoshka / tile / project.
- Persistence: auto-save with atomic write (tmp + rename), auto-load on init,
  `threading.RLock` around all mutations.
- **MCP server** with 9 tools — works with Claude Desktop / Claude Code / Cursor.
- **LangChain v1.0+ integration** via `CrystalMessageHistory`
  (`BaseChatMessageHistory` subclass). Session isolation, retrieval-style memory.
- **LlamaIndex** stub via `CrystalMemBuffer`.
- Benchmark scripts: episodic recall, capacity / scaling-law, realistic
  conversation (LongMemEval-style), LoCoMo multi-session, operation latency,
  Mem0-style semantic comparison.
- Demo scripts: personal AI, coding agent, team-shared pool, GDPR audit,
  portability cross-instance, memory marketplace.
- Tests covering all public APIs.
- Empirical validations:
  - Episodic recall: crystal_h16 vs mem0_buffer at N=1000 → **0.76 vs 0.02 (38× lift)**.
  - Recall scaling law: empirical R² = **0.968**.
  - Federated merge precision: empirical Δ = **8.6×10⁻⁸**.
- Documentation: README (EN/RU), QUICKSTART, API reference, CONTRIBUTING,
  CODE_OF_CONDUCT, LICENSE (MIT).
