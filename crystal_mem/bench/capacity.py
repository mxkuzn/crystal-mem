"""Empirical cap50 + recall law validation.

Validates two structural properties of block-octonion HRR memory:
    1. Capacity bound `cap50 = dim/16` per head — sharp phase transition.
    2. Recall scaling: `recall ≈ 1 − exp(−α·h·dim/N)` with α ≈ 0.139.

Method:
    For dim ∈ {256, 512, 1024} and h ∈ {1, 4, 16}:
      Sweep N from 4 to 4·cap50, measure recall_via_crystal() top-1 accuracy
      across 50 queries. Fit predicted vs observed.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from crystal_mem import CrystalMem
from crystal_mem.embedders import hash_embedder


def measure_recall(dim: int, h: int, N: int, eval_n: int = 50, seed: int = 0) -> float:
    e = hash_embedder(target_dim=dim)
    m = CrystalMem(dim=dim, n_heads=h, embedder=e)
    ids = [m.add(f"fact-{i}") for i in range(N)]

    g = torch.Generator().manual_seed(seed + 1)
    eval_idx = torch.randperm(N, generator=g)[: min(eval_n, N)].tolist()
    correct = 0
    for i in eval_idx:
        target_id = ids[i]
        entry, _ = m.recall_via_crystal(target_id)
        if entry and entry.id == target_id:
            correct += 1
    return correct / max(1, len(eval_idx))


def predicted_recall(N: int, h: int, dim: int, alpha: float = 0.139) -> float:
    return 1.0 - math.exp(-alpha * h * dim / max(N, 1))


def run_cap_validation(
    dims: tuple[int, ...] = (256, 512, 1024),
    heads: tuple[int, ...] = (1, 4, 16),
    eval_n: int = 50,
) -> dict:
    print("=" * 80)
    print("Capacity validation — recall law: 1 − exp(−0.139·h·dim/N)")
    print("=" * 80)
    print(f"\n{'dim':>6} {'h':>4} {'N':>6} {'cap50':>6} "
          f"{'observed':>10} {'predicted':>10} {'Δ':>8}")
    results = {}
    for dim in dims:
        cap50 = dim // 16
        sizes = [4, cap50 // 2, cap50, cap50 * 2, cap50 * 4]
        for h in heads:
            for N in sizes:
                t0 = time.perf_counter()
                obs = measure_recall(dim, h, N, eval_n=eval_n)
                pred = predicted_recall(N, h, dim)
                delta = obs - pred
                key = f"dim{dim}_h{h}_N{N}"
                results[key] = {
                    "dim": dim, "h": h, "N": N, "cap50": cap50,
                    "observed_recall": obs, "predicted_recall": pred,
                    "delta": delta, "wallclock_s": time.perf_counter() - t0,
                }
                print(f"{dim:>6} {h:>4} {N:>6} {cap50:>6} "
                      f"{obs:>10.3f} {pred:>10.3f} {delta:>+8.3f}")
        print()

    # R² between observed and predicted
    obs = [r["observed_recall"] for r in results.values()]
    pred = [r["predicted_recall"] for r in results.values()]
    obs_mean = sum(obs) / len(obs)
    ss_tot = sum((o - obs_mean) ** 2 for o in obs)
    ss_res = sum((o - p) ** 2 for o, p in zip(obs, pred))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    print(f"Total points: {len(obs)}")
    print(f"R² of observed vs predicted (recall law): {r2:.3f}")
    results["_r_squared"] = r2
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dims", default="256,512,1024")
    p.add_argument("--heads", default="1,4,16")
    p.add_argument("--eval-n", type=int, default=50)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    res = run_cap_validation(
        dims=tuple(int(x) for x in args.dims.split(",")),
        heads=tuple(int(x) for x in args.heads.split(",")),
        eval_n=args.eval_n,
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
