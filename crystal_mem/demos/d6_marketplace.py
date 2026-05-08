"""DEMO 6 — Memory marketplace (NEW APPLICATION).

Scenario: communities publish curated knowledge crystals as portable artifacts.
A new agent "subscribes" to multiple crystals — their facts auto-merge into
local memory.

Examples of marketable crystals:
    - "Python idioms 2026" — community-curated patterns
    - "GDPR compliance facts EU 2026"
    - "OpenSSL CVE summaries"
    - "Cooking technique reference (vegan)"

This is enabled uniquely by:
    1. Federated merge — no conflict resolution needed.
    2. Selective filtered export — publishers control what's in their crystal.
    3. Watermarking — provenance proof (see `watermark.py`).
    4. Exact unmerge — subscribers can revoke at will.

This NEW use case isn't possible with vector-DB-based memory systems because
they require centralized index re-training when sources change.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from crystal_mem import CrystalMem
from crystal_mem.embedders import sentence_transformer_embedder


def make_publisher_crystal(name: str, facts: list[str], embedder, dim=384):
    """A publisher curates a crystal and exports it for distribution."""
    m = CrystalMem(dim=dim, n_heads=4, embedder=embedder, user_id=f"pub:{name}")
    for f in facts:
        m.add(f, entity=name, tags={"published", name},
              metadata={"publisher": name, "version": "1.0"})
    return m.export()


def main():
    print("=" * 70)
    print("DEMO 6 — Memory marketplace (NEW use case)")
    print("=" * 70)

    embedder = sentence_transformer_embedder(target_dim=384)
    workdir = Path(tempfile.mkdtemp(prefix="cm_market_"))

    # === Publishers create curated crystals ===
    py_idioms = make_publisher_crystal(
        "python_idioms",
        [
            "Use list comprehensions over filter+map in Python",
            "Prefer pathlib over os.path for path operations",
            "Use dataclasses with @dataclass(frozen=True) for value objects",
            "Avoid mutable default arguments (use None + create inside fn)",
            "Use itertools.chain.from_iterable instead of sum(..., [])",
        ],
        embedder,
    )
    py_idioms.to_file(workdir / "python_idioms.crystal")

    gdpr_facts = make_publisher_crystal(
        "gdpr_2026",
        [
            "Article 17 grants right to erasure of personal data",
            "Data Processing Agreements required with all processors",
            "DPO appointment required for >250 employees",
            "Breach notification within 72 hours to supervisory authority",
            "Data Portability under Article 20 — provide in machine-readable format",
        ],
        embedder,
    )
    gdpr_facts.to_file(workdir / "gdpr_2026.crystal")

    rust_perf = make_publisher_crystal(
        "rust_perf",
        [
            "Use Box<[T]> instead of Vec<T> for read-only owned slices",
            "Prefer iterators over collect+for_each",
            "tokio::spawn_blocking for CPU-heavy in async context",
            "criterion benchmarks should warm up for 3 seconds minimum",
        ],
        embedder,
    )
    rust_perf.to_file(workdir / "rust_perf.crystal")

    # === User subscribes to relevant crystals ===
    print("\nUser subscribes to 3 published crystals...")
    me = CrystalMem(dim=384, n_heads=4, embedder=embedder, user_id="alice")

    me.add("My personal preference: 4-space indent",
           entity="alice", tags={"personal"})

    for name in ("python_idioms", "gdpr_2026", "rust_perf"):
        me.merge_file(workdir / f"{name}.crystal")

    print(f"  total memories: {len(me.entries)}")
    print(f"  entities: {sorted(me.entities())}")

    # Search across published + personal
    print("\n  Search 'how to handle paths in Python':")
    for e, s in me.search("how to handle paths in Python", top_k=2):
        print(f"    [{s:+.3f}] [{e.metadata.get('publisher', 'self')}] {e.content}")

    print("\n  Search 'data deletion law':")
    for e, s in me.search("data deletion law", top_k=2):
        print(f"    [{s:+.3f}] [{e.metadata.get('publisher', 'self')}] {e.content}")

    # === Publisher pushes update — new version ===
    print("\n--- Publisher 'python_idioms' v2.0 ships ---")
    py_idioms_v2 = make_publisher_crystal(
        "python_idioms",
        [
            # Updated content
            "Use match-case instead of long if-elif (Python 3.10+)",
            "Prefer | over Optional[T] (Python 3.10+)",
            "Use TypedDict over plain dict for structured data",
        ] + [
            # Original facts updated
            "Use list comprehensions over filter+map in Python",
            "Prefer pathlib over os.path for path operations",
        ],
        embedder,
    )

    # Subscriber: revoke v1, install v2
    me.unmerge(py_idioms)
    me.merge(py_idioms_v2)
    print(f"  unmerged v1, merged v2 — total memories: {len(me.entries)}")

    # === Trust model — selective subscription ===
    print("\n--- Selective subscription: only certain tags ---")
    fresh = CrystalMem(dim=384, n_heads=4, embedder=embedder, user_id="bob")

    # Bob only wants gdpr facts that mention "Article 17"
    art17_only = CrystalMem.from_crystal(gdpr_facts, embedder=embedder)
    art17_filter = art17_only.export(filter=lambda e: "Article 17" in e.content)
    fresh.merge(art17_filter)
    print(f"  Bob's filtered subscription: {len(fresh.entries)} entries")
    for e in fresh.get_all():
        print(f"    - {e.content}")

    shutil.rmtree(workdir)
    print("\nDONE")


if __name__ == "__main__":
    main()
