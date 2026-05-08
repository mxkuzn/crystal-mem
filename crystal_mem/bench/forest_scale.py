"""Forest scale test — does auto-clustering + compaction actually help at N=5K?

Compares:
    flat_crystal     single CrystalMem with N entries (above cap50)
    forest           CrystalForest with target=64, max=256
    forest_compact   forest after one compact() pass
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from crystal_mem import CrystalMem
from crystal_mem.embedders import sentence_transformer_embedder
from crystal_mem.forest import CrystalForest


CLUSTER_TEMPLATES = [
    ("Python", ["Python is dynamic", "Python uses indentation",
                "list comprehensions in Python", "Python type hints",
                "Python async coroutines"]),
    ("Cooking", ["recipe with garlic", "boil pasta al dente",
                 "marinate meat overnight", "use fresh herbs",
                 "balance sweet and salty"]),
    ("Travel", ["pack lightly for trips", "book flights early",
                "carry travel insurance", "learn local greetings",
                "try local food"]),
    ("Finance", ["track expenses monthly", "diversify investments",
                 "emergency fund matters", "compound interest works",
                 "tax-advantaged accounts"]),
    ("ML", ["regularization helps overfitting", "use cross-validation",
            "feature engineering matters", "watch for data leakage",
            "ensemble models often win"]),
]


def make_facts(n: int, seed: int = 0) -> list[tuple[str, str]]:
    """Returns list of (content, cluster_label)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        ci = int(torch.randint(0, len(CLUSTER_TEMPLATES), (1,), generator=g))
        cluster, base = CLUSTER_TEMPLATES[ci]
        bi = int(torch.randint(0, len(base), (1,), generator=g))
        suffix = int(torch.randint(0, 100000, (1,), generator=g))
        out.append((f"{base[bi]} (instance {suffix})", cluster))
    return out


def run(n: int = 5000, dim: int = 384, target: int = 64, max_per: int = 256):
    print("=" * 80)
    print(f"CrystalForest scale benchmark | N={n} | dim={dim} | "
          f"target={target} | max={max_per}")
    print("=" * 80)

    embedder = sentence_transformer_embedder(target_dim=dim)
    facts = make_facts(n)

    # Flat CrystalMem (above cap50)
    flat = CrystalMem(dim=dim, n_heads=4, embedder=embedder)
    t0 = time.perf_counter()
    for content, _ in facts:
        flat.add(content)
    t_flat_ingest = time.perf_counter() - t0
    print(f"\nflat_crystal:    {len(flat.entries)} entries, "
          f"ingest {t_flat_ingest:.1f}s, cap_used={flat.capacity_used:.0%}")

    # Forest
    forest = CrystalForest(dim=dim, target_per_crystal=target,
                           max_per_crystal=max_per, embedder=embedder)
    t0 = time.perf_counter()
    for content, cluster in facts:
        forest.add(content, entity=cluster)
    t_forest_ingest = time.perf_counter() - t0
    s = forest.stats()
    print(f"forest:          {s['total_entries']} entries, "
          f"ingest {t_forest_ingest:.1f}s, "
          f"shards={s['n_shards']}, mean_size={s['shard_size_mean']:.0f}, "
          f"min/max={s['shard_size_min']}/{s['shard_size_max']}, "
          f"cap_used_avg={s['avg_capacity_used']:.0%}")

    t0 = time.perf_counter()
    res = forest.compact()
    t_compact = time.perf_counter() - t0
    s = forest.stats()
    print(f"forest_compact:  ran in {t_compact*1000:.0f}ms, "
          f"shards {res['initial_shards']}→{res['final_shards']}, "
          f"merged={res['merged_pairs']}")

    # Search benchmark
    print("\n--- Search latency (50 queries) ---")
    queries = ["Python features I should know", "good cooking technique",
               "travel preparation tips", "financial advice for beginners",
               "machine learning best practices"] * 10

    t0 = time.perf_counter()
    [flat.search(q, top_k=5) for q in queries]
    t_flat_search = time.perf_counter() - t0
    print(f"flat_crystal:    {t_flat_search*1000/len(queries):.2f} ms/query")

    t0 = time.perf_counter()
    [forest.search(q, top_k=5, max_shards_to_probe=4) for q in queries]
    t_forest_search = time.perf_counter() - t0
    print(f"forest (4 probes): {t_forest_search*1000/len(queries):.2f} ms/query")

    t0 = time.perf_counter()
    [forest.search(q, top_k=5, max_shards_to_probe=999) for q in queries]
    t_forest_full = time.perf_counter() - t0
    print(f"forest (all):     {t_forest_full*1000/len(queries):.2f} ms/query")

    # Cluster purity check — how often does forest route correctly?
    print("\n--- Cluster purity ---")
    purity = []
    for sid, shard in forest.shards.items():
        clusters = []
        for e in shard.memory.entries.values():
            ent = e.metadata.get("entity")
            if ent:
                clusters.append(ent)
        if clusters:
            from collections import Counter
            counts = Counter(clusters)
            purity_score = counts.most_common(1)[0][1] / len(clusters)
            purity.append(purity_score)
    if purity:
        avg_purity = sum(purity) / len(purity)
        print(f"avg shard cluster purity: {avg_purity:.0%} "
              f"(min={min(purity):.0%}, max={max(purity):.0%})")
        print("  (1.0 = perfect routing; 0.2 = random)")

    return {
        "flat_ingest_s": t_flat_ingest, "forest_ingest_s": t_forest_ingest,
        "compact_ms": t_compact * 1000,
        "flat_search_ms": t_flat_search * 1000 / len(queries),
        "forest_search_ms": t_forest_search * 1000 / len(queries),
        "n_shards": s["n_shards"],
        "avg_purity": sum(purity) / len(purity) if purity else 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--target", type=int, default=64)
    p.add_argument("--max", type=int, default=256)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    res = run(n=args.n, dim=args.dim, target=args.target, max_per=args.max)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
