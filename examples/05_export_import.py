"""05 — portable crystal files: export, share, import.

A `.crystal` file is a self-contained binary snapshot. It can travel between
machines / instances / processes. The format is stable and forward-compatible.
"""
import tempfile
from pathlib import Path
from crystal_mem import CrystalMem


# Build a crystal
laptop = CrystalMem(dim=512, n_heads=2, user_id="alice")
laptop.add("Project Phoenix uses event sourcing", tags={"work"})
laptop.add("Wife's birthday is October 5",        tags={"personal"})
laptop.add("Allergic to shellfish",               tags={"medical"})

with tempfile.TemporaryDirectory() as td:
    crystal_path = Path(td) / "alice.crystal"

    # Export full memory
    laptop.export_file(crystal_path)
    print(f"exported {crystal_path.stat().st_size:,} bytes")

    # Import on another machine / instance
    phone = CrystalMem.from_file(crystal_path)
    print(f"phone loaded {len(phone.entries)} memories")

    # Export a *filtered subset* — only "work" facts. Useful for sharing
    # team conventions without leaking personal data.
    work_path = Path(td) / "alice_work_only.crystal"
    laptop.export_file(work_path, filter=lambda e: "work" in e.tags)

    new_hire = CrystalMem.from_file(work_path)
    print("\nnew_hire imported only the work-tagged subset:")
    for e in new_hire.get_all():
        print(f"  - {e.content}  (tags: {sorted(e.tags)})")

    # Diff between two memories
    only_in_a, only_in_b, common = laptop.diff(new_hire)
    print(f"\ndiff: only in laptop = {len(only_in_a)}, only in new_hire = {len(only_in_b)}")
