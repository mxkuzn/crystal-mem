"""CrystalMem — research-backed agent memory with portability and exact forget.

Public API:
    Memory       — main class (formerly named CrystalMem; alias kept for clarity).
    CrystalMem   — alias for Memory.
    MemoryEntry  — single memory record.
    Crystal      — exported / imported binary container.

Quick start:
    >>> from crystal_mem import CrystalMem
    >>> m = CrystalMem(user_id="alice", dim=1024, mode="explicit")
    >>> m.add("I prefer Python over Go for backend")
    >>> m.search("language preferences", top_k=3)
"""
from __future__ import annotations

from .core import CrystalMem, Memory, MemoryEntry, ProvenanceStep
from .portability import Crystal

__version__ = "0.1.0"
__all__ = [
    "CrystalMem", "Memory", "MemoryEntry", "ProvenanceStep", "Crystal",
    "__version__",
]
