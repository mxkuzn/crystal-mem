# Quickstart — 5 minutes from zero to running

**Read in:** [English](QUICKSTART.md) · [Русский](QUICKSTART.ru.md)

## Install

```bash
# Core
pip install crystal-mem

# With real semantic embeddings (recommended)
pip install crystal-mem[embeddings]

# With everything (MCP server, LangChain, LlamaIndex, extractors)
pip install crystal-mem[all]
```

## 30-second tour

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

# Semantic search
for entry, score in m.search("what languages do I like?", top_k=2):
    print(f"  [{score:+.3f}] {entry.content}")

# GDPR-grade exact forget (math-clean, no reindex)
m.forget_by_filter(lambda e: "medical" in e.tags)
```

## Common tasks

### Persist across runs

```python
m = CrystalMem(
    user_id="alice", dim=384, n_heads=4,
    persist_path="/path/to/alice.crystal",   # auto-save on every mutation
)
```

### Per-entity scoping (knowledge-graph mode)

```python
m.add("uses Postgres", entity="atlas")
m.add("uses Tokio runtime", entity="atlas")
m.add("uses Vue 3", entity="beacon")

m.entity_facts("atlas")                                 # all atlas facts
m.search_by_entity("atlas", "what database?", top_k=2)  # entity-scoped search

m.link("alice", "acme", "works_at")                     # KG edges
m.link("bob",   "acme", "works_at")
m.entity_facts("acme")                                  # both alice & bob bound
```

### Memory portability

```python
m.export_file("alice.crystal")                          # serialize to disk

m2 = CrystalMem.from_file("alice.crystal")              # load on another machine

m2.merge_file("partner.crystal")                        # federated combine
m2.unmerge_file("merged_by_mistake.crystal")            # reverse a merge

# Selective export — only "work" facts
m.export_file("work.crystal", filter=lambda e: "work" in e.tags)
```

### Plug into LangChain

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

### Migrate from another memory tool

```python
from crystal_mem.migrate import migrate_from_mem0

m = CrystalMem(user_id="alice", dim=384)
n = migrate_from_mem0(m, "mem0_dump.json")
print(f"imported {n} memories")
```

CLI equivalent:

```bash
python -m crystal_mem.migrate \
    --source mem0 --input dump.json \
    --output alice.crystal --dim 384
```

### Run the MCP server (Claude Desktop / Cursor / etc.)

```bash
python -m crystal_mem.mcp_server \
    --user-id alice --dim 1024 \
    --state ~/.crystal_mem/alice.crystal
```

Add to Claude Desktop config:

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

Tools exposed: `memory_add`, `memory_search`, `memory_get`, `memory_forget`,
`memory_list`, `memory_export`, `memory_merge_file`, `memory_unmerge_file`,
`memory_stats`.

## Next

- [API reference](API.md) — every public function with one-line description.
- [`../examples/`](../examples/) — 10 single-purpose snippets.
- [`../crystal_mem/demos/`](../crystal_mem/demos/) — 6 full scenarios, including the
  [memory marketplace](../crystal_mem/demos/d6_marketplace.py).
- [Capacity guide](../examples/10_capacity_awareness.py) — when to scale dim
  vs heads vs forest.
