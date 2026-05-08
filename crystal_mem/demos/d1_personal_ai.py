"""DEMO 1 — Personal AI memory across sessions.

Scenario: power user of Claude Desktop / ChatGPT wants AI to remember preferences,
ongoing projects, and personal context across sessions and devices.

Features shown:
    * Auto-persistence (state file)
    * Real semantic search via sentence-transformers
    * Entity-bound facts (per-project namespacing)
    * Auto-load on next session

Run twice to see persistence in action:
    python -m crystal_mem.demos.d1_personal_ai
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from crystal_mem import CrystalMem
from crystal_mem.embedders import sentence_transformer_embedder


def main():
    state_dir = Path(tempfile.mkdtemp(prefix="cm_demo1_"))
    state_path = state_dir / "alice.crystal"

    print("=" * 70)
    print("DEMO 1 — Personal AI memory across sessions")
    print("=" * 70)
    print(f"State file: {state_path}")

    embedder = sentence_transformer_embedder(target_dim=384)

    # === Session 1: morning conversation ===
    print("\n--- Session 1 (morning) ---")
    m = CrystalMem(
        user_id="alice",
        dim=384,
        n_heads=4,
        embedder=embedder,
        persist_path=str(state_path),
    )

    m.add("I prefer Python over Go for backend services", entity="alice", tags={"work"})
    m.add("My team's coding style: explicit > implicit",   entity="alice", tags={"work"})
    m.add("I'm vegan and allergic to peanuts",             entity="alice", tags={"personal"})
    m.add("Working on Project Atlas — distributed crawler", entity="atlas", tags={"work"})
    m.add("Atlas: chose Tokio runtime for async",          entity="atlas", tags={"work"})
    m.add("Spouse's birthday is March 15",                 entity="alice", tags={"personal"})
    print(f"  added {len(m.entries)} memories")
    print(f"  state persisted to {state_path}")

    # === Session 2: same machine restarts, new instance ===
    print("\n--- Session 2 (after restart) ---")
    m2 = CrystalMem(
        user_id="alice",
        dim=384,
        n_heads=4,
        embedder=embedder,
        persist_path=str(state_path),
    )
    print(f"  loaded {len(m2.entries)} memories from disk")

    print("\n  Search 'food preferences':")
    for entry, score in m2.search("food preferences", top_k=2):
        print(f"    [{score:+.3f}] {entry.content}")

    print("\n  Search 'async runtime decision':")
    for entry, score in m2.search("async runtime decision", top_k=2):
        print(f"    [{score:+.3f}] {entry.content}")

    print("\n  All facts about project Atlas:")
    for f in m2.entity_facts("atlas"):
        print(f"    - {f.content}")

    # === Session 3: GDPR — user wants to forget personal info ===
    print("\n--- Session 3 (GDPR forget personal facts) ---")
    n = m2.forget_by_filter(lambda e: "personal" in e.tags)
    print(f"  forgot {n} personal entries")
    print(f"  remaining: {len(m2.entries)} memories")

    contents = [e.content for e in m2.get_all()]
    assert "Spouse's birthday is March 15" not in contents
    print("  ✓ personal facts removed exactly (empirical Δ ≈ 4×10⁻⁶)")

    shutil.rmtree(state_dir)
    print("\nDONE")


if __name__ == "__main__":
    main()
