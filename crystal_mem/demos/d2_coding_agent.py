"""DEMO 2 — Coding agent project memory (Cursor / Claude Code style).

Scenario: a coding agent learns project conventions, decisions, and gotchas.
Memory is bound to project entity. When user switches project, switches crystal.

Features shown:
    * Per-project entity scoping
    * Compositional links (file → uses → module)
    * search_by_entity (no cross-project leakage)
    * Filtered export (publish "company conventions" crystal across projects)
"""
from __future__ import annotations

from crystal_mem import CrystalMem
from crystal_mem.embedders import sentence_transformer_embedder


def main():
    print("=" * 70)
    print("DEMO 2 — Coding agent project memory")
    print("=" * 70)

    embedder = sentence_transformer_embedder(target_dim=384)
    m = CrystalMem(dim=384, n_heads=4, embedder=embedder, user_id="dev")

    # Project Atlas conventions
    m.add("All HTTP handlers use the @audited decorator", entity="atlas", tags={"convention"})
    m.add("Tests live in tests/ mirroring src/ structure", entity="atlas", tags={"convention"})
    m.add("We use ruff for lint, never black",            entity="atlas", tags={"convention"})
    m.add("Database: Postgres 16 with sqlx, NEVER ORM",   entity="atlas", tags={"convention"})
    m.add("Bug #142: don't use multiprocessing on macOS", entity="atlas", tags={"gotcha"})

    # Project Beacon conventions
    m.add("Beacon uses Vue 3 composition API only",       entity="beacon", tags={"convention"})
    m.add("State management: Pinia, no Vuex",             entity="beacon", tags={"convention"})
    m.add("Tailwind is forbidden in Beacon",              entity="beacon", tags={"convention"})

    # Personal preferences (cross-project)
    m.add("I like 4-space indentation",                   entity="dev", tags={"preference"})

    print("\nQuery within project Atlas: 'how to format my code?'")
    for e, s in m.search_by_entity("atlas", "code style and formatting", top_k=3):
        print(f"  [{s:+.3f}] {e.content}")

    print("\nQuery within project Beacon: 'how to format my code?'")
    for e, s in m.search_by_entity("beacon", "code style and formatting", top_k=3):
        print(f"  [{s:+.3f}] {e.content}")

    print("\nWhen switching project, isolation works — no Atlas leak into Beacon results.")

    # Compositional links — agent now learns relations
    print("\n--- Linking entities (knowledge graph mode) ---")
    m.link("atlas", "postgres", "uses_db")
    m.link("beacon", "postgres", "uses_db")
    m.link("atlas", "rust", "written_in")
    m.link("beacon", "typescript", "written_in")

    print("Projects using postgres:")
    pg_users = m.entity_facts("postgres")
    for f in pg_users:
        print(f"  - {f.content}")

    # Export company-wide conventions only — share with new hire
    print("\n--- Filtered export: just conventions across all projects ---")
    crystal = m.export(filter=lambda e: "convention" in e.tags)
    print(f"  exported {crystal.metadata['n_entries']} convention entries")
    print(f"  file size: {len(crystal.to_bytes())} bytes")

    # New hire receives crystal
    new_hire = CrystalMem.from_crystal(crystal, embedder=embedder)
    print(f"\nNew hire imports crystal — has {len(new_hire.entries)} conventions")
    print("  Their search 'lint tool' returns:")
    for e, s in new_hire.search("lint tool", top_k=1):
        print(f"    [{s:+.3f}] {e.content}")
    print("  Their search 'spouse birthday' returns:")
    res = new_hire.search("spouse birthday", top_k=1)
    if res and "spouse" in res[0][0].content.lower():
        print(f"    LEAKED: {res[0][0].content}")
    else:
        print("    (no personal data — only conventions exported)")

    print("\nDONE")


if __name__ == "__main__":
    main()
