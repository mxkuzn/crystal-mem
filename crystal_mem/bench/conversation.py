"""Realistic conversation benchmark — multi-turn LongMemEval-style.

Protocol:
    1. 100 turns of mixed conversation: user shares facts, asks questions, idle chitchat.
    2. Memory accumulates facts via two strategies (explicit + auto).
    3. After all turns, ask 30 needle-in-haystack questions referring to facts
       that may be 5..90 turns old.
    4. Compare:
        crystal_mem (explicit add)         — agent extracts via simple regex
        crystal_mem (h=4)                  — multi-head ensemble
        mem0_buffer (last K=20 verbatim)   — Mem0-style without LLM extractor
        bank_only (FAISS-flat ceiling)     — pure dense retrieval ceiling

Tests both semantic similarity (paraphrase) AND temporal recall (deep history).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass

import torch

from crystal_mem import CrystalMem
from crystal_mem.embedders import sentence_transformer_embedder


# -------------------------------------------------------------------------
# Synthetic dialog generator — needles + chitchat
# -------------------------------------------------------------------------
NEEDLE_TEMPLATES = [
    ("My favorite color is {x}", "what's my favorite color?", "{x}"),
    ("I work at {x}", "where do I work?", "{x}"),
    ("My birthday is on {x}", "when is my birthday?", "{x}"),
    ("I have a {x} cat named {y}", "describe my cat", "{x} cat named {y}"),
    ("I drive a {x}", "what car do I drive?", "{x}"),
    ("My favorite cuisine is {x}", "what food do I prefer?", "{x}"),
    ("I'm allergic to {x}", "do I have any allergies?", "{x}"),
    ("My emergency contact is {x}", "who's my emergency contact?", "{x}"),
    ("My pet peeve is {x}", "what really annoys me?", "{x}"),
    ("My password reset secret is {x}", "what's my recovery secret?", "{x}"),
]

NOUN_VAR = ["azure", "Acme Corp", "March 5th", "tabby", "Max", "Tesla 3",
            "Italian", "peanuts", "my sister Lila", "loud chewing", "violet-432-stream",
            "sage green", "Pixel Corp", "October 12", "tortie", "Whiskers",
            "Volvo XC40", "Japanese", "shellfish", "my friend Tomas",
            "interrupting", "sunset-9712-flame"]

CHITCHAT = [
    "How's the weather today?", "Did you see the game last night?",
    "Mondays are tough.", "Coffee is fuel.", "Working from home today.",
    "Just had lunch.", "Need a vacation.", "Traffic was bad.",
    "Heard about the news?", "Quick question for you.",
]


@dataclass
class Turn:
    role: str
    content: str
    needle_idx: int | None = None        # if a needle, which one
    needle_value: str | None = None      # the answer payload


def make_dialogue(n_turns: int = 100, n_needles: int = 30, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    needles = []  # list of (turn_idx, fact, query, answer)
    turns: list[Turn] = []

    needle_positions = sorted(torch.randperm(n_turns - 5, generator=g)[:n_needles].tolist())
    set(needle_positions)
    needle_iter = iter(needle_positions)
    next_needle = next(needle_iter, None)

    for t in range(n_turns):
        if t == next_needle:
            template = NEEDLE_TEMPLATES[t % len(NEEDLE_TEMPLATES)]
            fact_t, query_t, ans_t = template
            x = NOUN_VAR[int(torch.randint(0, len(NOUN_VAR), (1,), generator=g))]
            y = NOUN_VAR[int(torch.randint(0, len(NOUN_VAR), (1,), generator=g))]
            fact = fact_t.format(x=x, y=y)
            answer = ans_t.format(x=x, y=y)
            turns.append(Turn(role="user", content=fact,
                              needle_idx=len(needles), needle_value=answer))
            needles.append((t, fact, query_t, answer))
            next_needle = next(needle_iter, None)
        else:
            cc = CHITCHAT[int(torch.randint(0, len(CHITCHAT), (1,), generator=g))]
            turns.append(Turn(role="user", content=cc))

    return turns, needles


# -------------------------------------------------------------------------
# Fact extractor (cheap heuristic — no LLM)
# -------------------------------------------------------------------------
FACT_PATTERNS = [
    re.compile(r"^my (.+?) is (.+)$", re.IGNORECASE),
    re.compile(r"^i (work at|drive|love|prefer|hate|am allergic to|have a) (.+)$", re.IGNORECASE),
    re.compile(r"^(.+?) is on (.+)$", re.IGNORECASE),
]


def is_fact_like(text: str) -> bool:
    return any(p.match(text.strip()) for p in FACT_PATTERNS)


# -------------------------------------------------------------------------
# Method evaluators
# -------------------------------------------------------------------------
def eval_crystal_mem(turns, needles, embedder, dim: int, n_heads: int):
    m = CrystalMem(dim=dim, n_heads=n_heads, embedder=embedder)
    t0 = time.perf_counter()
    for turn in turns:
        if is_fact_like(turn.content):
            m.add(turn.content)
    t_ingest = time.perf_counter() - t0

    correct = 0
    for _, fact, query, answer in needles:
        results = m.search(query, top_k=1)
        if results and answer.lower() in results[0][0].content.lower():
            correct += 1
    return correct / len(needles), t_ingest, len(m.entries)


def eval_mem0_buffer(turns, needles, embedder, dim: int, K: int = 20):
    """Last-K-turns verbatim baseline (≈ Mem0 without LLM)."""
    facts = []
    for turn in turns:
        if is_fact_like(turn.content):
            facts.append(turn.content)
    bank = facts[-K:] if len(facts) > K else facts
    if not bank:
        return 0.0, 0.0, 0
    t0 = time.perf_counter()
    bank_emb = torch.stack([embedder(f) for f in bank])
    bank_n = bank_emb / bank_emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    t_ingest = time.perf_counter() - t0

    correct = 0
    for _, fact, query, answer in needles:
        q = embedder(query)
        qn = q / q.norm().clamp_min(1e-12)
        sims = qn.unsqueeze(0) @ bank_n.t()
        top = int(sims.squeeze(0).argmax().item())
        if answer.lower() in bank[top].lower():
            correct += 1
    return correct / len(needles), t_ingest, len(bank)


def eval_bank_only(turns, needles, embedder, dim: int):
    """FAISS-flat ceiling — keep ALL facts as embeddings."""
    facts = [t.content for t in turns if is_fact_like(t.content)]
    if not facts:
        return 0.0, 0.0, 0
    t0 = time.perf_counter()
    bank_emb = torch.stack([embedder(f) for f in facts])
    bank_n = bank_emb / bank_emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    t_ingest = time.perf_counter() - t0

    correct = 0
    for _, fact, query, answer in needles:
        q = embedder(query)
        qn = q / q.norm().clamp_min(1e-12)
        sims = qn.unsqueeze(0) @ bank_n.t()
        top = int(sims.squeeze(0).argmax().item())
        if answer.lower() in facts[top].lower():
            correct += 1
    return correct / len(needles), t_ingest, len(facts)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def run(n_turns: int = 100, n_needles: int = 30, dim: int = 384, seed: int = 0):
    print("=" * 80)
    print(f"Conversation benchmark | turns={n_turns}, needles={n_needles}, dim={dim}")
    print("=" * 80)

    embedder = sentence_transformer_embedder(target_dim=dim)
    turns, needles = make_dialogue(n_turns, n_needles, seed)

    methods = [
        ("mem0_buffer (K=20)", lambda: eval_mem0_buffer(turns, needles, embedder, dim, K=20)),
        ("bank_only (ceiling)", lambda: eval_bank_only(turns, needles, embedder, dim)),
        ("crystal_mem (h=1)", lambda: eval_crystal_mem(turns, needles, embedder, dim, n_heads=1)),
        ("crystal_mem (h=4)", lambda: eval_crystal_mem(turns, needles, embedder, dim, n_heads=4)),
    ]

    print(f"\n{'method':<26} {'recall':>8} {'ingest_s':>10} {'stored':>8}")
    results = {}
    for name, fn in methods:
        acc, t_ingest, stored = fn()
        results[name] = {"recall": acc, "ingest_s": t_ingest, "stored": stored}
        print(f"{name:<26} {acc:>8.3f} {t_ingest:>10.2f} {stored:>8}")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--turns", type=int, default=100)
    p.add_argument("--needles", type=int, default=30)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    res = run(n_turns=args.turns, n_needles=args.needles, dim=args.dim)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
