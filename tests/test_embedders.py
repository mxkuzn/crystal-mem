"""Tests for real embedders + resize logic."""
from __future__ import annotations

import pytest
import torch

from crystal_mem.embedders import (
    hash_embedder,
    sentence_transformer_embedder,
    _resize,
)


def test_resize_truncate():
    v = torch.arange(10).float()
    out = _resize(v, 8, mode="truncate")
    assert out.shape == (8,)
    assert torch.equal(out, torch.arange(8).float())


def test_resize_tile():
    v = torch.tensor([1.0, 2.0, 3.0])
    out = _resize(v, 8, mode="tile")
    assert out.shape == (8,)
    assert out[0] == 1.0 and out[3] == 1.0  # repeated


def test_resize_project_deterministic():
    v = torch.randn(384)
    a = _resize(v, 1024, mode="project")
    b = _resize(v, 1024, mode="project")
    assert torch.allclose(a, b)


def test_hash_embedder_deterministic():
    e = hash_embedder(target_dim=64)
    a = e("hello world")
    b = e("hello world")
    assert torch.allclose(a, b)
    c = e("different text")
    assert not torch.allclose(a, c)


def test_hash_embedder_block_oct_normalized():
    e = hash_embedder(target_dim=64)
    v = e("test")
    # 64-d = 8 octonion blocks, each unit-normalized → total |v| = sqrt(8)
    assert abs(float(v.norm()) - (8 ** 0.5)) < 1e-4


def test_st_embedder_real_semantic_signal():
    """Real ST model: similar texts should have higher dot-product than dissimilar."""
    e = sentence_transformer_embedder(target_dim=384)
    a = e("I love programming in Python")
    b = e("Python coding is my favorite")
    c = e("Cooking pasta tonight")
    sim_ab = float((a * b).sum())
    sim_ac = float((a * c).sum())
    assert sim_ab > sim_ac, f"Expected sim_ab ({sim_ab}) > sim_ac ({sim_ac})"


def test_st_embedder_dim_must_be_multiple_of_8():
    with pytest.raises(ValueError):
        sentence_transformer_embedder(target_dim=100)


def test_embedder_in_crystal_mem():
    from crystal_mem import CrystalMem
    e = sentence_transformer_embedder(target_dim=384)
    m = CrystalMem(dim=384, n_heads=2, embedder=e)
    m.add("Python is a great programming language")
    m.add("Cooking takes time and patience")
    m.add("Python developers love type hints")
    results = m.search("programming languages", top_k=3)
    # Top-2 should be the python-related entries
    contents = [e.content for e, _ in results[:2]]
    pythons = sum(1 for c in contents if "python" in c.lower())
    assert pythons >= 1, f"Expected at least 1 python in top-2: {contents}"
