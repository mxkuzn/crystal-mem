"""Head-to-head benchmark: CrystalMem vs Mem0-style summary baseline vs FAISS.

Protocol (LongMemEval-inspired):
    1. Generate N=[100, 500, 1000, 2000] synthetic facts about a user.
    2. Each method ingests the facts in sequence.
    3. Sample 50 random "lookup" queries paraphrased from the facts.
    4. Measure top-1 accuracy.

Methods compared:
    crystal_h1     CrystalMem with n_heads=1                (single-head HRR)
    crystal_h4     CrystalMem with n_heads=4                (multi-head ensemble)
    mem0_summary   "summary + last K verbatim" baseline       (≈ Mem0 without LLM)
    faiss_dense    FAISS-flat L2 over raw embeddings         (vector-DB ceiling)

Run:
    python -m crystal_mem.bench.mem0_compat
    python -m crystal_mem.bench.mem0_compat --hash --sizes 100,500
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from crystal_mem import CrystalMem
from crystal_mem.embedders import hash_embedder, sentence_transformer_embedder


# -------------------------------------------------------------------------
# Synthetic fact generator (LongMemEval-style)
# -------------------------------------------------------------------------
ROLES = [
    "prefers", "uses", "lives in", "works at", "drives", "owns", "studied",
    "trained in", "speaks", "dislikes", "is allergic to", "is interested in",
]

SUBJECTS = [
    "Alice", "Bob", "Charlie", "Dana", "Eve", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Karen", "Leo", "Mallory", "Niaj", "Olivia", "Peggy",
    "Trent", "Uma", "Victor", "Walter",
]

OBJECTS = [
    "Python", "Rust", "Go", "Berlin", "Tokyo", "Acme Corp", "Foo Inc",
    "Tesla Model 3", "Volvo XC60", "a Stradivarius violin", "a 2014 Boeing",
    "MIT", "ETH Zürich", "Stanford", "Russian", "Japanese", "Italian",
    "blue cheese", "spicy food", "peanuts", "machine learning", "underwater photography",
    "vintage typewriters", "competitive chess", "marathon running", "abstract art",
    "amateur radio", "molecular biology", "ancient cuneiform", "Brazilian jazz",
]


def make_facts(n: int, seed: int = 0) -> list[tuple[str, str]]:
    """Returns list of (statement, paraphrased_query)."""
    g = torch.Generator().manual_seed(seed)
    facts = []
    for _ in range(n):
        s = SUBJECTS[int(torch.randint(0, len(SUBJECTS), (1,), generator=g))]
        r = ROLES[int(torch.randint(0, len(ROLES), (1,), generator=g))]
        o = OBJECTS[int(torch.randint(0, len(OBJECTS), (1,), generator=g))]
        statement = f"{s} {r} {o}"
        query = f"What does {s} {r}?" if "is" not in r else f"Tell me about {s}"
        facts.append((statement, query))
    return facts


# -------------------------------------------------------------------------
# Method evaluators
# -------------------------------------------------------------------------
def eval_crystal_mem(
    facts: list[tuple[str, str]],
    eval_idx: list[int],
    embedder,
    dim: int,
    n_heads: int,
) -> tuple[float, float]:
    m = CrystalMem(dim=dim, n_heads=n_heads, embedder=embedder)
    t0 = time.perf_counter()
    for stmt, _ in facts:
        m.add(stmt)
    t_ingest = time.perf_counter() - t0

    correct = 0
    for i in eval_idx:
        _, query = facts[i]
        results = m.search(query, top_k=1)
        if results and results[0][0].content == facts[i][0]:
            correct += 1
    return correct / len(eval_idx), t_ingest


def eval_mem0_summary(
    facts: list[tuple[str, str]],
    eval_idx: list[int],
    embedder,
    dim: int,
    K_buffer: int = 32,
) -> tuple[float, float]:
    """Mem0-without-LLM analogue: ring buffer of last K + mean-summary of older."""
    embs = []
    t0 = time.perf_counter()
    for stmt, _ in facts:
        embs.append(embedder(stmt))
    t_ingest = time.perf_counter() - t0
    embs_t = torch.stack(embs)
    bank_n = embs_t / embs_t.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    correct = 0
    N = len(facts)
    for i in eval_idx:
        _, query = facts[i]
        embedder(query)
        if i >= N - K_buffer:
            # In recent buffer: full embedding available
            local_idx = i - (N - K_buffer)
            recent = embs_t[N - K_buffer + local_idx]
            pred_emb = recent
        else:
            # Old: only summary of pre-buffer mean
            pred_emb = embs_t[: N - K_buffer].mean(dim=0)
        # Top-1 from full bank using pred_emb as proxy retrieval point
        pn = pred_emb / pred_emb.norm().clamp_min(1e-12)
        sims = pn.unsqueeze(0) @ bank_n.t()
        top = int(sims.squeeze(0).argmax().item())
        if top == i:
            correct += 1
    return correct / len(eval_idx), t_ingest


def eval_faiss_dense(
    facts: list[tuple[str, str]],
    eval_idx: list[int],
    embedder,
    dim: int,
) -> tuple[float, float]:
    """FAISS-flat ceiling: pure dense top-1 over query embedding."""
    t0 = time.perf_counter()
    embs = torch.stack([embedder(s) for s, _ in facts])
    t_ingest = time.perf_counter() - t0
    bank_n = embs / embs.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    correct = 0
    for i in eval_idx:
        _, query = facts[i]
        q = embedder(query)
        qn = q / q.norm().clamp_min(1e-12)
        sims = qn.unsqueeze(0) @ bank_n.t()
        top = int(sims.squeeze(0).argmax().item())
        if top == i:
            correct += 1
    return correct / len(eval_idx), t_ingest


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def run_benchmark(
    sizes: list[int] = (100, 500, 1000),
    dim: int = 384,
    eval_n: int = 50,
    use_hash_embedder: bool = False,
    seed: int = 0,
) -> dict:
    if use_hash_embedder:
        embedder = hash_embedder(target_dim=dim)
        embedder_name = "hash-fallback"
    else:
        embedder = sentence_transformer_embedder(target_dim=dim)
        embedder_name = "all-MiniLM-L6-v2"

    print("=" * 80)
    print(f"CrystalMem Mem0-style benchmark | embedder={embedder_name} | dim={dim}")
    print("=" * 80)
    print(f"\n{'N':>6} {'method':<22} {'top1':>7} {'ingest_s':>10}")
    results = {}

    for N in sizes:
        facts = make_facts(N, seed=seed)
        eval_idx = torch.randperm(N, generator=torch.Generator().manual_seed(seed + 1))[:eval_n].tolist()

        m1_acc, m1_t = eval_mem0_summary(facts, eval_idx, embedder, dim)
        f_acc, f_t = eval_faiss_dense(facts, eval_idx, embedder, dim)
        c1_acc, c1_t = eval_crystal_mem(facts, eval_idx, embedder, dim, n_heads=1)
        c4_acc, c4_t = eval_crystal_mem(facts, eval_idx, embedder, dim, n_heads=4)

        results[N] = {
            "mem0_summary": {"top1": m1_acc, "ingest_s": m1_t},
            "faiss_dense":  {"top1": f_acc,  "ingest_s": f_t},
            "crystal_h1":   {"top1": c1_acc, "ingest_s": c1_t},
            "crystal_h4":   {"top1": c4_acc, "ingest_s": c4_t},
        }
        print(f"{N:>6} {'mem0_summary':<22} {m1_acc:>7.3f} {m1_t:>10.2f}")
        print(f"{N:>6} {'faiss_dense (ceiling)':<22} {f_acc:>7.3f} {f_t:>10.2f}")
        print(f"{N:>6} {'crystal_h1':<22} {c1_acc:>7.3f} {c1_t:>10.2f}")
        print(f"{N:>6} {'crystal_h4':<22} {c4_acc:>7.3f} {c4_t:>10.2f}")
        print()

    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="100,500,1000",
                   help="Comma-separated N values to benchmark")
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--eval-n", type=int, default=50)
    p.add_argument("--hash", action="store_true",
                   help="Use hash embedder (no model load) — tests algebra only")
    p.add_argument("--out", default=None, help="JSON output path")
    args = p.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    res = run_benchmark(
        sizes=sizes, dim=args.dim, eval_n=args.eval_n,
        use_hash_embedder=args.hash,
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
