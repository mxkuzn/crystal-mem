"""03 — persistence: auto-save and auto-load.

Pass `persist_path=...` and CrystalMem writes the full state to that file
atomically on every mutation (add, update, forget). Next process that points
at the same path loads automatically.

Use for: long-running agents, IDE plugins, MCP servers, anything that
should survive a restart.
"""
import tempfile
from pathlib import Path
from crystal_mem import CrystalMem


with tempfile.TemporaryDirectory() as td:
    state = Path(td) / "alice.crystal"

    # Session 1 — fresh memory, writes to disk
    m1 = CrystalMem(dim=512, n_heads=2, persist_path=str(state))
    m1.add("ate breakfast at 8am")
    m1.add("scheduled meeting at 3pm")
    print(f"session 1 wrote {state.stat().st_size:,} bytes")

    # Session 2 — new instance, same path, auto-loads
    m2 = CrystalMem(dim=512, n_heads=2, persist_path=str(state))
    print(f"session 2 loaded {len(m2.entries)} memories from disk")
    for entry in m2.get_all():
        print(f"  - {entry.content}")

    # Forget mutates AND persists
    m2.forget(list(m2.entries.keys())[0])
    print(f"after forget: {len(m2.entries)} (re-load and you'll see the same)")
