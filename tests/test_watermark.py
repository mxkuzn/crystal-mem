"""Watermark tests — z-score detection."""
from __future__ import annotations

from crystal_mem import CrystalMem
from crystal_mem.watermark import watermark_inject, watermark_detect


def test_watermark_injection_detected():
    m = CrystalMem(dim=512, n_heads=4)
    for i in range(20):
        m.add(f"fact {i}")
    watermark_inject(m, label="acme-corp")
    res = watermark_detect(m, label="acme-corp")
    assert res.detected, f"Expected detection, got {res}"
    assert res.z_score > res.threshold


def test_no_false_positive_other_label():
    m = CrystalMem(dim=512, n_heads=4)
    for i in range(20):
        m.add(f"fact {i}")
    watermark_inject(m, label="acme-corp")
    res = watermark_detect(m, label="evilcorp-stolen")
    # Different label — should NOT detect
    assert not res.detected, f"False positive: {res}"


def test_watermark_survives_export_import():
    m = CrystalMem(dim=512, n_heads=4)
    for i in range(20):
        m.add(f"fact {i}")
    watermark_inject(m, label="published_v1")

    crystal = m.export()
    res = watermark_detect(crystal, label="published_v1")
    assert res.detected, f"Watermark lost after export: {res}"


def test_watermark_survives_merge():
    """Even after merge into another instance, watermark stays detectable."""
    publisher = CrystalMem(dim=512, n_heads=4)
    for i in range(15):
        publisher.add(f"published fact {i}")
    watermark_inject(publisher, label="acme_pub_v1")

    user = CrystalMem(dim=512, n_heads=4)
    user.add("user's own fact")
    user.merge(publisher)

    res = watermark_detect(user, label="acme_pub_v1")
    assert res.detected, f"Watermark lost after merge: {res}"


def test_no_watermark_no_detection():
    """Empty / unwatermarked crystal should not trigger detection."""
    m = CrystalMem(dim=512, n_heads=4)
    for i in range(20):
        m.add(f"fact {i}")
    # No injection
    res = watermark_detect(m, label="any_label")
    assert not res.detected, f"False positive on clean crystal: {res}"
