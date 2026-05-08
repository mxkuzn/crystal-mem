"""Watermarking helpers — z-score scheme with FP < 0.003%.

Watermark a crystal with a deterministic signature derivable from a secret label.
Detection works on any crystal — even after merge/unmerge — because the watermark
contribution is preserved by HRR linearity.

Method:
    1. From label + salt, derive K random "carrier" octonion vectors.
    2. Inject γ × Σ carriers into each head's crystal (γ scales detection threshold).
    3. To detect: compute correlation between unknown crystal and same K carriers.
       Statistic z = mean(<crystal, c_i>) / sigma — should be γ·√K vs random √K.

For γ ≈ √N (N = num entries), false-positive rate < 0.003%
(reproducible from `tests/test_watermark.py`).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ._octonion_ops import block_oct_normalize

if TYPE_CHECKING:
    from .core import CrystalMem


@dataclass
class WatermarkResult:
    label: str
    z_score: float                    # higher = more confident watermark present
    threshold: float                  # decision threshold (typically 4.0 = 0.003% FP)
    detected: bool

    def __repr__(self) -> str:
        return (
            f"<Watermark label={self.label!r} "
            f"z={self.z_score:.2f} thr={self.threshold:.1f} "
            f"detected={self.detected}>"
        )


def _carriers(label: str, dim: int, n_carriers: int = 32, salt: str = "") -> torch.Tensor:
    """Deterministic carriers for a given label."""
    out = []
    for k in range(n_carriers):
        digest = hashlib.sha256(f"{salt}|{label}|wm-c{k}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        g = torch.Generator().manual_seed(seed)
        out.append(block_oct_normalize(torch.randn(dim, generator=g)))
    return torch.stack(out)


def watermark_inject(
    memory: "CrystalMem",
    label: str,
    *,
    gamma: float | None = None,
    n_carriers: int = 32,
    salt: str = "",
) -> None:
    """Inject a watermark into all heads of `memory`.

    gamma controls strength. If None, defaults to sqrt(max(n_entries, 16)).
    """
    n_entries = max(len(memory.entries), 16)
    g = gamma if gamma is not None else (n_entries ** 0.5)
    carriers = _carriers(label, memory.dim, n_carriers, salt=salt)
    contrib = g * carriers.sum(dim=0)
    with memory._lock:
        for h in range(memory.n_heads):
            memory.crystals[h] = memory.crystals[h] + contrib
        memory._persist_if_configured()


def watermark_detect(
    crystal_or_memory,
    label: str,
    *,
    n_carriers: int = 32,
    salt: str = "",
    threshold: float = 4.0,
) -> WatermarkResult:
    """Detect watermark in a CrystalMem instance OR a portable Crystal.

    Uses the average of signed projections across heads + carriers; threshold
    z=4.0 corresponds to FP<0.003%.
    """
    # Acquire crystals + dim (either from CrystalMem or Crystal)
    if hasattr(crystal_or_memory, "crystals") and hasattr(crystal_or_memory, "dim"):
        crystals = crystal_or_memory.crystals
        dim = crystal_or_memory.dim
    else:  # portable Crystal
        crystals = crystal_or_memory.crystals
        dim = crystal_or_memory.metadata["dim"]

    carriers = _carriers(label, dim, n_carriers, salt=salt)
    # carriers: [K, dim]. crystals: list of [dim].
    crystal_stack = torch.stack(crystals)            # [H, dim]
    proj = crystal_stack @ carriers.t()              # [H, K]
    # Z-score: mean over (H, K) projections, divided by std of random projections.
    # For random unit-norm vectors of dim d, projection has std ≈ sqrt(K/d) * |M|.
    # We approximate via empirical std across (H*K) values themselves.
    flat = proj.flatten()
    mean = float(flat.mean())
    std = float(flat.std().clamp_min(1e-12))
    z = abs(mean) / std * (flat.numel() ** 0.5)
    return WatermarkResult(
        label=label, z_score=z, threshold=threshold,
        detected=z > threshold,
    )
