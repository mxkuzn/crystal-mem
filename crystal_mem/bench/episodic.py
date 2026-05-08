"""Episodic / key-based recall benchmark.

The headline scenario:
    The agent receives a stream of N events. Later, asked to recall what
    happened at event K, where K may be deep in the past.

Mem0-style buffer: only last K events stored verbatim, older are mean-summarized.
Beyond the buffer, recall is essentially random.

CrystalMem (HRR): each event bound to its turn-key. Recall via unbind(M, key).
Multi-head ensemble lifts capacity along the empirical recall scaling law.

This benchmark **does not** require semantic similarity between query and fact —
it requires exact key recall. Hence different protocol from `mem0_compat.py`.
"""
from __future__ import annotations

import argparse
import json

import torch

from crystal_mem._octonion_ops import bind, block_oct_conj, block_oct_normalize


def run_episodic(
    sizes: tuple[int, ...] = (100, 500, 1000, 2000),
    dim: int = 1024,
    n_heads_list: tuple[int, ...] = (1, 4, 16),
    eval_n: int = 50,
    K_buffer: int = 32,
    seed: int = 0,
) -> dict:
    """Random-embedding episodic benchmark — no semantic component.

    Measures pure capacity of the memory representation.
    """
    print("=" * 80)
    print(f"Episodic recall benchmark | dim={dim}")
    print("=" * 80)
    print(f"\n{'N':>6} {'method':<28} {'recall':>8}")

    results = {}

    for N in sizes:
        g = torch.Generator().manual_seed(seed)
        # Each event: a "turn key" + a "fact embedding".
        block_oct_normalize(torch.randn(N, dim, generator=g))
        fact_emb = block_oct_normalize(torch.randn(N, dim, generator=g))
        bn = fact_emb / fact_emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        eval_idx = torch.randperm(N, generator=g)[:eval_n].tolist()

        results[N] = {}

        # Mem0-style: ring buffer + summary
        recent_facts = fact_emb[-K_buffer:] if N >= K_buffer else fact_emb
        summary = (
            fact_emb[: N - K_buffer].mean(dim=0)
            if N > K_buffer
            else torch.zeros(dim)
        )
        correct = 0
        for i in eval_idx:
            if i >= N - K_buffer:
                idx_local = i - (N - K_buffer)
                pred = recent_facts[idx_local]
            else:
                pred = summary
            pn = pred / pred.norm().clamp_min(1e-12)
            top = int((pn.unsqueeze(0) @ bn.t()).argmax().item())
            if top == i:
                correct += 1
        recall_mem0 = correct / eval_n
        results[N]["mem0_buffer"] = recall_mem0
        print(f"{N:>6} {'mem0_buffer (K=32)':<28} {recall_mem0:>8.3f}")

        # CrystalMem (HRR ensemble) — different head counts
        for h in n_heads_list:
            crystals = []
            head_keys_list = []
            for he in range(h):
                kh = block_oct_normalize(
                    torch.randn(N, dim, generator=torch.Generator().manual_seed(seed + 100 + he))
                )
                crystals.append(bind(kh, fact_emb).sum(dim=0))
                head_keys_list.append(kh)
            correct = 0
            for i in eval_idx:
                outs = []
                for he in range(h):
                    r = bind(
                        block_oct_conj(head_keys_list[he][i:i+1]),
                        crystals[he].unsqueeze(0),
                    ).squeeze(0)
                    outs.append(r)
                pred = torch.stack(outs, dim=0).mean(dim=0)
                pn = pred / pred.norm().clamp_min(1e-12)
                top = int((pn.unsqueeze(0) @ bn.t()).argmax().item())
                if top == i:
                    correct += 1
            recall_h = correct / eval_n
            results[N][f"crystal_h{h}"] = recall_h
            print(f"{N:>6} {f'crystal_h{h}':<28} {recall_h:>8.3f}")
        print()

    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="100,500,1000,2000")
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--heads", default="1,4,16")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    sizes = tuple(int(x) for x in args.sizes.split(","))
    heads = tuple(int(x) for x in args.heads.split(","))
    res = run_episodic(sizes=sizes, dim=args.dim, n_heads_list=heads)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
