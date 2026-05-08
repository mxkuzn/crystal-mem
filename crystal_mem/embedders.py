"""Real embedders for CrystalMem.

Default: sentence-transformers/all-MiniLM-L6-v2 (384-d) projected to target dim.
Optional: Qwen/Qwen3-Embedding-0.6B (1024-d Matryoshka — natively truncatable to 64..1024).

The embedder is a callable `(str) -> Tensor[dim]` returning a *block-octonion-normalized*
vector ready for write into a CrystalMem crystal (dim must be multiple of 8).

Resize strategies when source-dim ≠ target-dim:
    "truncate"  — take first `dim` coords (works for Matryoshka models like Qwen3).
    "project"   — random fixed projection to dim (deterministic, seeded).
    "tile"      — tile + average (default for non-Matryoshka small models).
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Callable, Literal

import torch

from ._octonion_ops import block_oct_normalize


ResizeMode = Literal["truncate", "project", "tile", "auto"]


def _resize(vec: torch.Tensor, target_dim: int, mode: ResizeMode = "auto") -> torch.Tensor:
    src = vec.shape[-1]
    if src == target_dim:
        return vec
    if mode == "auto":
        mode = "truncate" if src >= target_dim else "tile"

    if mode == "truncate":
        if src < target_dim:
            return _resize(vec, target_dim, mode="tile")
        return vec[..., :target_dim]
    if mode == "tile":
        n_repeat = (target_dim + src - 1) // src
        tiled = vec.repeat(*([1] * (vec.dim() - 1)), n_repeat)
        return tiled[..., :target_dim]
    if mode == "project":
        g = torch.Generator().manual_seed(0xC0FFEE)
        proj = torch.randn(src, target_dim, generator=g) / (src ** 0.5)
        return vec @ proj
    raise ValueError(f"Unknown resize mode: {mode}")


@lru_cache(maxsize=4)
def _load_st_model(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def sentence_transformer_embedder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    target_dim: int = 384,
    resize_mode: ResizeMode = "auto",
    normalize_block_oct: bool = True,
) -> Callable[[str], torch.Tensor]:
    """Returns an embedder backed by sentence-transformers.

    Default model: all-MiniLM-L6-v2 (384-d, fast, runs on CPU).
    Pad target_dim to multiple of 8 for block-oct compatibility.
    """
    if target_dim % 8 != 0:
        raise ValueError("target_dim must be multiple of 8 for block-octonion algebra")

    model = _load_st_model(model_name)

    def _embed(text: str) -> torch.Tensor:
        vec = torch.tensor(
            model.encode(text, convert_to_numpy=True, show_progress_bar=False),
            dtype=torch.float32,
        )
        vec = _resize(vec, target_dim, mode=resize_mode)
        if normalize_block_oct:
            vec = block_oct_normalize(vec)
        return vec

    _embed.dim = target_dim  # type: ignore[attr-defined]
    _embed.source_model = model_name  # type: ignore[attr-defined]
    return _embed


def qwen3_embedder(
    target_dim: int = 1024,
    resize_mode: ResizeMode = "truncate",
) -> Callable[[str], torch.Tensor]:
    """Qwen3-Embedding-0.6B (1024-d Matryoshka)."""
    return sentence_transformer_embedder(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        target_dim=target_dim,
        resize_mode=resize_mode,
    )


def hash_embedder(target_dim: int = 1024) -> Callable[[str], torch.Tensor]:
    """Deterministic hash-based fallback. Useful for testing without model load."""
    if target_dim % 8 != 0:
        raise ValueError("target_dim must be multiple of 8")

    def _embed(text: str) -> torch.Tensor:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        g = torch.Generator().manual_seed(seed)
        return block_oct_normalize(torch.randn(target_dim, generator=g))

    _embed.dim = target_dim  # type: ignore[attr-defined]
    _embed.source_model = "sha256-hash"  # type: ignore[attr-defined]
    return _embed


def get_default_embedder(target_dim: int = 1024) -> Callable[[str], torch.Tensor]:
    """Try to return a real embedder; fall back to hash if no model installed."""
    try:
        return sentence_transformer_embedder(target_dim=target_dim)
    except Exception:
        return hash_embedder(target_dim=target_dim)
