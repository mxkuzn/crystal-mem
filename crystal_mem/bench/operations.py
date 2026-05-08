"""Operation latency benchmark — forget, merge, export, load.

These are the operations where CrystalMem has unique value:
    forget       : exact algebraic subtract (vs vector DB rebuild or filter)
    merge        : commutative sum (vs vector DB index re-train)
    export/load  : binary serialization (vs vector DB checkpoint)

Compares against a naive "list of embeddings" baseline (≈ what Mem0 / lightweight
vector DB do under the hood).
"""
from __future__ import annotations

import argparse
import json
import time


from crystal_mem import CrystalMem
from crystal_mem.embedders import hash_embedder


def bench_naive_forget(N: int, dim: int) -> float:
    """Naive: list of embeddings, forget = list.remove + reindex hash table."""
    e = hash_embedder(target_dim=dim)
    facts = [f"fact {i}" for i in range(N)]
    embs = [e(f) for f in facts]
    ids = list(range(N))
    id_to_idx = {i: idx for idx, i in enumerate(ids)}

    t0 = time.perf_counter()
    for i in range(0, N, max(1, N // 100)):  # 100 forgets
        idx = id_to_idx.get(i)
        if idx is None:
            continue
        # remove + rebuild idx (this is what most vector dbs end up doing)
        del embs[idx]
        del ids[idx]
        id_to_idx = {i_: idx_ for idx_, i_ in enumerate(ids)}
    return (time.perf_counter() - t0) / 100


def bench_crystal_forget(N: int, dim: int) -> float:
    e = hash_embedder(target_dim=dim)
    m = CrystalMem(dim=dim, n_heads=2, embedder=e)
    mids = [m.add(f"fact {i}") for i in range(N)]

    t0 = time.perf_counter()
    for mid in mids[: max(1, N // 100) * 100 : max(1, N // 100)]:
        m.forget(mid)
    return (time.perf_counter() - t0) / 100


def bench_naive_merge(N: int, dim: int) -> float:
    e = hash_embedder(target_dim=dim)
    a = [e(f"a {i}") for i in range(N)]
    b = [e(f"b {i}") for i in range(N)]
    t0 = time.perf_counter()
    a + b  # naive list concat — hard to do for HNSW etc
    return time.perf_counter() - t0


def bench_crystal_merge(N: int, dim: int) -> float:
    e = hash_embedder(target_dim=dim)
    m1 = CrystalMem(dim=dim, n_heads=2, embedder=e)
    m2 = CrystalMem(dim=dim, n_heads=2, embedder=e)
    for i in range(N):
        m1.add(f"a {i}")
        m2.add(f"b {i}")
    t0 = time.perf_counter()
    m1.merge(m2)
    return time.perf_counter() - t0


def bench_crystal_export_load(N: int, dim: int) -> tuple[float, float, int]:
    import tempfile
    from pathlib import Path
    e = hash_embedder(target_dim=dim)
    m = CrystalMem(dim=dim, n_heads=2, embedder=e)
    for i in range(N):
        m.add(f"fact {i}")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "x.crystal"
        t0 = time.perf_counter()
        m.export_file(path)
        t_export = time.perf_counter() - t0

        t0 = time.perf_counter()
        CrystalMem.from_file(path, embedder=e)
        t_load = time.perf_counter() - t0
        size = path.stat().st_size
    return t_export, t_load, size


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="100,500,1000")
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    print("=" * 80)
    print(f"Operation latency | dim={args.dim}")
    print("=" * 80)
    print(f"\n{'N':>6} {'op':<22} {'time_ms':>10} {'extra':>14}")

    results = {}
    for N in sizes:
        results[N] = {}

        t = bench_crystal_forget(N, args.dim)
        results[N]["crystal_forget_per_op_ms"] = t * 1000
        print(f"{N:>6} {'crystal forget/op':<22} {t*1000:>10.3f}")

        t = bench_naive_forget(N, args.dim)
        results[N]["naive_forget_per_op_ms"] = t * 1000
        print(f"{N:>6} {'naive list forget/op':<22} {t*1000:>10.3f}")

        t = bench_crystal_merge(N, args.dim)
        results[N]["crystal_merge_total_ms"] = t * 1000
        print(f"{N:>6} {'crystal merge total':<22} {t*1000:>10.3f}")

        t = bench_naive_merge(N, args.dim)
        results[N]["naive_merge_total_ms"] = t * 1000
        print(f"{N:>6} {'naive concat merge':<22} {t*1000:>10.3f}")

        t_e, t_l, sz = bench_crystal_export_load(N, args.dim)
        results[N]["export_ms"] = t_e * 1000
        results[N]["load_ms"] = t_l * 1000
        results[N]["file_size_bytes"] = sz
        print(f"{N:>6} {'export':<22} {t_e*1000:>10.3f} {sz:>10}b")
        print(f"{N:>6} {'load':<22} {t_l*1000:>10.3f}")
        print()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
