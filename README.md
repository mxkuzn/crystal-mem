# CrystalMem

**Read in:** [English](README.md) · [Русский](README.ru.md)

> Research-backed agent memory with mathematically exact GDPR-clean forget,
> federated merge, and portable crystal files.

[![PyPI](https://img.shields.io/pypi/v/crystal-mem.svg)](https://pypi.org/project/crystal-mem/)
[![Python](https://img.shields.io/pypi/pyversions/crystal-mem.svg)](https://pypi.org/project/crystal-mem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/mxkuzn/crystal-mem/actions/workflows/test.yml/badge.svg)](https://github.com/mxkuzn/crystal-mem/actions/workflows/test.yml)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Quick links:**
[Quickstart](docs/QUICKSTART.md) ·
[API reference](docs/API.md) ·
[Examples](examples/) ·
[Demos](crystal_mem/demos/) ·
[Changelog](CHANGELOG.md) ·
[Use-cases](USE_CASES.md)

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

## Why

LLM agents (ChatGPT, Claude, Cursor, Devin, Letta) forget everything beyond the
context window. Existing memory libs (Mem0, MemGPT, Zep, GraphMemory) lose recall
fast at scale and cannot truly delete information for GDPR / AI Act compliance.

**CrystalMem** uses block-octonion HRR (Holographic Reduced Representations).
Empirical numbers — all reproducible from `crystal_mem/bench/`:

| Property | Number | Reproduce |
|---|---|---|
| Recall vs Mem0-buffer at N=1000 (episodic) | **0.76 vs 0.02 (38× lift)** | `crystal_mem/bench/episodic.py` |
| CoT scratchpad on 50-step chains | **100%** | internal test |
| Exact forget Δ\|M\|/\|M\| | **4×10⁻⁶** | `tests/test_core.py::test_forget_subtracts_from_crystal_exactly` |
| Federated merge identical to centralized | Δ = **8.6×10⁻⁸** | `crystal_mem/demos/d3_team_shared_pool.py` |
| Watermark FP rate | **<0.003%** | `tests/test_watermark.py` |
| Capacity bound | cap50 = dim / 16 | `crystal_mem/bench/capacity.py` |
| Recall scaling law (empirical R²) | **0.968** | `crystal_mem/bench/capacity.py` |
| Compositional bind precision | cos > 0.98 | internal test |

## Install

```bash
pip install crystal-mem                    # core
pip install crystal-mem[embeddings]        # + sentence-transformers / Qwen3
pip install crystal-mem[mcp]               # + MCP server for Claude Desktop / Cursor
pip install crystal-mem[langchain]         # + CrystalMessageHistory adapter
pip install crystal-mem[llamaindex]
pip install crystal-mem[all,dev]           # everything + test tools
```

## 30-second demo

```python
from crystal_mem import CrystalMem
from crystal_mem.embedders import sentence_transformer_embedder

embed = sentence_transformer_embedder(target_dim=384)
m = CrystalMem(user_id="alice", dim=384, n_heads=4, embedder=embed,
               persist_path="alice.crystal")     # auto-save on every mutation

m.add("I prefer Python over Go", entity="alice", tags={"work"})
m.add("Working on Project Atlas — distributed crawler", entity="atlas")
m.add("Allergic to peanuts", entity="alice", tags={"medical"})

for entry, score in m.search("language preferences", top_k=2):
    print(f"  [{score:+.3f}] {entry.content}")

# GDPR-clean forget (math-exact, no reindex)
m.forget_entity("alice")
```

## Two operating modes

```python
# Mode 1 — explicit (default, no LLM cost)
m = CrystalMem(mode="explicit")
m.add("user prefers vegan food")

# Mode 2 — auto (extracts facts via LLM, Mem0-compatible ergonomics)
from crystal_mem.extractors import openai_extractor
m = CrystalMem(mode="auto", llm_extractor=openai_extractor("gpt-4o-mini"))
m.add_from_conversation([
    {"role": "user", "content": "I'm vegan, just FYI"},
    {"role": "assistant", "content": "Got it!"},
])
```

## GDPR-grade forget

```python
m.forget(memory_id)                                       # single entry
m.forget_entity("customer-c001")                          # all about an entity
m.forget_by_filter(lambda e: e.user_id == "alice")        # bulk
```

The crystal returns to its mathematical pre-add state (Δ ≈ 4×10⁻⁶). Not
a soft-delete, not a filter — algebraically exact subtraction.

## Memory portability — take your AI memory anywhere

```python
m.export_file("alice.crystal")                # binary serialization

m2 = CrystalMem.from_file("alice.crystal")    # load on another machine

# Merge (mathematically identical to centralized rebuild — Δ ≈ 9×10⁻⁸)
m_combined = CrystalMem(dim=1024, n_heads=4)
m_combined.merge_file("alice_work.crystal")
m_combined.merge_file("alice_personal.crystal")

# Selective export — only "work"-tagged entries
m.export_file("work.crystal", filter=lambda e: "work" in e.tags)

# Diff between two memories
only_a, only_b, common = m_a.diff(m_b)

# Undo a merge
m.unmerge_file("merged_in_by_mistake.crystal")
```

## Entity-bound memory (knowledge-graph mode)

```python
m.add("uses Postgres", entity="atlas")
m.add("uses Tokio runtime", entity="atlas")
m.add("uses Vue 3", entity="beacon")

m.entity_facts("atlas")                      # all facts about Atlas
m.search_by_entity("atlas", "what database?", top_k=2)

# Bidirectional links — KG-style
m.link("alice", "acme", "works_at")
m.link("alice", "berlin", "lives_in")
m.entity_facts("acme")                       # alice works_at acme
```

## MCP server (Claude Desktop / Claude Code / Cursor)

```bash
python -m crystal_mem.mcp_server \
    --user-id alice --dim 1024 --heads 4 \
    --state ~/.crystal_mem/alice.crystal
```

Wire into Claude Desktop config:

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

9 tools exposed: `memory_add`, `memory_search`, `memory_get`, `memory_forget`,
`memory_list`, `memory_export`, `memory_merge_file`, `memory_unmerge_file`,
`memory_stats`.

## LangChain integration

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

Drop-in for `langchain-core >= 1.0`. Session-isolated, retrieval-style memory
optionally available.

## Migration from Mem0 / Letta / Zep

```python
from crystal_mem.migrate import migrate_from_mem0, migrate_from_letta

m = CrystalMem(user_id="alice", dim=384)
n = migrate_from_mem0(m, "mem0_dump.json")
print(f"Imported {n} memories from Mem0")
```

CLI:
```bash
python -m crystal_mem.migrate \
    --source mem0 --input dump.json \
    --output alice.crystal --dim 384
```

Supports: `mem0`, `letta`, `zep`, `chat-jsonl`.

## NEW use case — Memory marketplace

Federated merge + watermarking + selective filter enables curated knowledge
crystals as **portable artifacts**:

```python
# A community publishes "Python idioms 2026"
publisher = CrystalMem(dim=384, n_heads=4)
for fact in PYTHON_IDIOMS:
    publisher.add(fact, tags={"published"})
watermark_inject(publisher, label="python_idioms_2026")
publisher.export_file("python_idioms.crystal")

# Subscribers merge — no re-index, no conflict resolution
me = CrystalMem(dim=384, n_heads=4)
me.merge_file("python_idioms.crystal")
me.merge_file("gdpr_compliance_2026.crystal")
me.merge_file("rust_perf_tips.crystal")

# Verify provenance
detect = watermark_detect(me, label="python_idioms_2026")
assert detect.detected            # source-of-truth verified
```

This is **uniquely** enabled by the federated-merge algebra — vector-DB-based
solutions cannot do this without re-training the index.

## Watermarking

```python
from crystal_mem.watermark import watermark_inject, watermark_detect

publisher = CrystalMem(...)
watermark_inject(publisher, label="acme_corp_v1")    # γ ≈ √N
publisher.export_file("acme_data.crystal")

# Anyone receiving the crystal can verify provenance
result = watermark_detect("acme_data.crystal", label="acme_corp_v1")
assert result.detected and result.z_score > 4.0
```

## Capacity awareness

```python
print(m.stats())
# {
#   "n_entries": 800, "dim": 1024, "n_heads": 4,
#   "cap50": 64, "capacity_used": 12.5,
#   "recall_estimate": 0.43,           # from empirical scaling law
#   ...
# }
print(m.capacity_warning())
# "Capacity exceeded: 800 entries vs cap50=64. Recall will degrade per
#  recall_law(N=800, h=4, dim=1024). Increase dim or n_heads."
```

## Benchmarks

Reproducible from the repo:

```bash
python -m crystal_mem.bench.episodic       # episodic recall — N up to 2000
python -m crystal_mem.bench.locomo         # multi-session LoCoMo-style
python -m crystal_mem.bench.conversation   # 100-300 turns realistic dialogue
python -m crystal_mem.bench.capacity       # capacity / scaling-law validation
python -m crystal_mem.bench.operations     # forget / merge / export latency
python -m crystal_mem.bench.mem0_compat    # CrystalMem vs FAISS vs Mem0-summary
```

Sample output (`bench.episodic`):
```
     N method                    recall
   500 mem0_buffer (K=32)         0.10
   500 crystal_h16                1.00      ← 10× lift
  1000 mem0_buffer (K=32)         0.02
  1000 crystal_h16                0.76      ← 38× lift
```

## Demos

```bash
python -m crystal_mem.demos.d1_personal_ai      # auto-persist + GDPR forget
python -m crystal_mem.demos.d2_coding_agent     # per-project memory
python -m crystal_mem.demos.d3_team_shared_pool # federated merge
python -m crystal_mem.demos.d4_gdpr_audit       # Article 17 deletion
python -m crystal_mem.demos.d5_portability      # cross-instance sync
python -m crystal_mem.demos.d6_marketplace      # publisher/subscriber crystals
```

## Documentation

| Where to look | What's there |
|---|---|
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | 5-minute getting-started |
| [`docs/API.md`](docs/API.md) | Every public function with one-line description |
| [`examples/`](examples/) | 10 single-purpose minimal scripts (20-60 lines each) |
| [`crystal_mem/demos/`](demos/) | 6 full end-to-end scenarios |
| [`USE_CASES.md`](USE_CASES.md) | Product-layer ideas built on top of the library |
| [`CRYSTAL_FOREST.md`](CRYSTAL_FOREST.md) | Experimental sharded forest (auto-cluster + KG hybrid) |
| [`CHANGELOG.md`](CHANGELOG.md) | Release notes |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | PR / issue process |
| [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) | Community guidelines |

## License

MIT — see [LICENSE](LICENSE).

## Citation

If CrystalMem helps your work:

```
@software{crystalmem,
  title  = {CrystalMem: agent memory with exact forget, federated merge, and portable crystal files},
  author = {Maxim Kuznetsov},
  year   = {2026},
  url    = {https://github.com/mxkuzn/crystal-mem}
}
```
