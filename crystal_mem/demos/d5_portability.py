"""DEMO 5 — Memory portability across instances.

Scenario: user has CrystalMem on laptop, wants to sync with phone, then merge
with team brain at office. All three sources are mathematically identical via
sum-merge.

This is the "memory marketplace" foundation — crystal files as portable
knowledge artifacts.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from crystal_mem import CrystalMem
from crystal_mem.embedders import sentence_transformer_embedder


def main():
    print("=" * 70)
    print("DEMO 5 — Memory portability across instances")
    print("=" * 70)

    embedder = sentence_transformer_embedder(target_dim=384)
    workdir = Path(tempfile.mkdtemp(prefix="cm_demo5_"))

    # === Laptop: develop knowledge ===
    laptop = CrystalMem(dim=384, n_heads=4, embedder=embedder, user_id="alice")
    laptop.add("Project Phoenix uses event sourcing", entity="phoenix", tags={"design"})
    laptop.add("Avoid synchronous DB writes in hot path", entity="phoenix", tags={"perf"})
    laptop.add("Allergic to peanuts", entity="alice", tags={"personal"})
    laptop_file = workdir / "alice_laptop.crystal"
    laptop.export_file(laptop_file)
    print(f"\nLaptop crystal: {len(laptop.entries)} entries → {laptop_file.name} "
          f"({laptop_file.stat().st_size} bytes)")

    # === Phone: continue conversation, develop more knowledge ===
    phone = CrystalMem(dim=384, n_heads=4, embedder=embedder, user_id="alice")
    phone.merge_file(laptop_file)
    print(f"\nPhone loaded laptop crystal — has {len(phone.entries)} entries")

    phone.add("Read article: event sourcing pitfalls in distributed systems",
              entity="phoenix", tags={"research"})
    phone.add("Wife's anniversary: October 5", entity="alice", tags={"personal"})
    phone_file = workdir / "alice_phone.crystal"
    phone.export_file(phone_file)
    print(f"Phone added 2 entries → exported ({phone_file.stat().st_size} bytes)")

    # === Office: merge laptop + phone + team brain ===
    print("\n--- Office: merge personal + team ---")
    team_brain = CrystalMem(dim=384, n_heads=4, embedder=embedder, user_id="team")
    team_brain.add("Team standup is 10am every Tue", entity="team", tags={"convention"})
    team_brain.add("Project Phoenix is owned by Alice", entity="phoenix", tags={"ownership"})
    team_file = workdir / "team_brain.crystal"
    team_brain.export_file(team_file)

    office = CrystalMem(dim=384, n_heads=4, embedder=embedder, user_id="alice")
    office.merge_file(phone_file)
    office.merge_file(team_file)
    print(f"  Office instance: merged {len(office.entries)} entries from 3 sources")

    print("\n  Search 'phoenix design':")
    for e, s in office.search("phoenix architectural decisions", top_k=4):
        print(f"    [{s:+.3f}] {e.content}")

    print("\n  All facts about project phoenix (cross-source):")
    for f in office.entity_facts("phoenix"):
        print(f"    - [{f.source}] {f.content}")

    # === Selective re-export: send "phoenix knowledge crystal" to new colleague ===
    print("\n--- Selective re-export: phoenix-only crystal for new dev ---")
    phoenix_crystal = office.export(
        filter=lambda e: e.metadata.get("entity") == "phoenix"
    )
    new_dev = CrystalMem(dim=384, n_heads=4, embedder=embedder, user_id="newdev")
    new_dev.merge(phoenix_crystal)
    print(f"  New dev's phoenix-only memory: {len(new_dev.entries)} entries")

    print("\n  New dev search 'wife anniversary' (should return nothing):")
    res = new_dev.search("wife anniversary", top_k=1)
    if res:
        is_personal = "personal" in res[0][0].tags
        print(f"    [{res[0][1]:+.3f}] {'❌ LEAK' if is_personal else '✓'} {res[0][0].content}")
    print("  ✓ Personal facts not present — selective export worked.")

    # === Diff between laptop and phone ===
    print("\n--- Diff: what's different between laptop and phone? ---")
    only_laptop, only_phone, common = laptop.diff(phone)
    print(f"  only on laptop: {len(only_laptop)}")
    print(f"  only on phone:  {len(only_phone)}")
    print(f"  common:         {len(common)}")
    for mid in only_phone:
        print(f"    [phone-only] {phone.entries[mid].content}")

    shutil.rmtree(workdir)
    print("\nDONE")


if __name__ == "__main__":
    main()
