"""LoCoMo-style multi-session benchmark.

Inspired by Maharana et al. 2024 ("Evaluating Very Long-Term Conversational
Memory of LLM Agents") — long conversations split across multiple sessions
with cross-session questions.

Synthetic to avoid LoCoMo dataset license. The protocol mirrors:
    1. K=5 sessions, each with ~30 turns of dialogue.
    2. Each session contains 3-4 needle facts.
    3. After all sessions, 50 questions asked, each referring to a fact from
       a *previous* session (not the current one).
    4. Method must recall the right session's needle.

Methods compared:
    - mem0_buffer (K=20)         most recent K verbatim
    - bank_only (FAISS-flat)     dense top-1 from full embedding bank
    - crystal_h1                 single-head HRR
    - crystal_h4                 multi-head HRR ensemble

The interesting comparison: when N_total_facts > buffer_K, mem0_buffer drops
old facts. CrystalMem keeps everything bound algebraically.
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


# Templates: (fact, query, expected_substring)
TEMPLATES = [
    ("On {date} I started learning {x}", "when did I start {x}?", "{date}"),
    ("My favorite {category} is {item}", "what's my favorite {category}?", "{item}"),
    ("I went to {place} last {when}", "where did I go {when}?", "{place}"),
    ("My {relation} works at {company}", "where does my {relation} work?", "{company}"),
    ("I bought a {product} for {price}", "what did I buy?", "{product}"),
    ("The {project} deadline is {date}", "when is the {project} deadline?", "{date}"),
    ("My doctor prescribed {medicine}", "what medicine was prescribed?", "{medicine}"),
    ("I scheduled {event} for {time}", "when is {event}?", "{time}"),
]

VARS = {
    "date":     ["March 5", "April 12", "May 30", "June 18", "September 3", "October 22"],
    "x":        ["Spanish", "guitar", "machine learning", "cooking", "kayaking", "stoicism"],
    "category": ["book", "movie", "drink", "song", "podcast"],
    "item":     ["Anna Karenina", "Inception", "matcha latte", "Bohemian Rhapsody", "Lex Fridman"],
    "place":    ["Lisbon", "Kyoto", "Cape Town", "Reykjavik", "Buenos Aires"],
    "when":     ["weekend", "month", "summer", "spring"],
    "relation": ["sister", "brother", "uncle", "cousin", "neighbor"],
    "company":  ["Acme Corp", "Foo Inc", "Bar Ltd", "Baz GmbH"],
    "product":  ["Stradivarius violin", "Volvo XC40", "espresso machine", "kayak"],
    "price":    ["$2400", "€18000", "$799", "$1100"],
    "project":  ["Phoenix", "Atlas", "Beacon", "Northstar"],
    "medicine": ["amoxicillin 500mg", "vitamin D3", "lisinopril 10mg", "cetirizine 10mg"],
    "event":    ["dentist appointment", "yoga class", "team standup"],
    "time":     ["Wednesday 2pm", "Friday morning", "Tuesday at noon"],
}

CHITCHAT = [
    "How are you doing today?", "Cold morning today.", "Coffee time.",
    "Stuck in traffic.", "Had pasta for lunch.", "Need to rest soon.",
    "What's up?", "Quick check.", "Working from cafe.", "Random thought.",
]


def render_needle(template_idx: int, rng: torch.Generator):
    fact_t, query_t, ans_t = TEMPLATES[template_idx]
    fields = re.findall(r"\{(\w+)\}", fact_t + query_t + ans_t)
    chosen = {}
    for f in set(fields):
        choices = VARS.get(f, [f])
        chosen[f] = choices[int(torch.randint(0, len(choices), (1,), generator=rng))]
    fact = fact_t.format(**chosen)
    query = query_t.format(**chosen)
    answer = ans_t.format(**chosen)
    return fact, query, answer


@dataclass
class SessionData:
    session_id: int
    turns: list[str]                  # all turns (chitchat + facts)
    needles: list[tuple[str, str, str]]  # list of (fact, query, expected_answer)


def make_locomo_sessions(n_sessions: int = 5, turns_per_session: int = 30,
                         needles_per_session: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    sessions = []
    for s in range(n_sessions):
        positions = sorted(torch.randperm(turns_per_session, generator=g)[:needles_per_session].tolist())
        pos_set = set(positions)
        turns: list[str] = []
        needles = []
        for t in range(turns_per_session):
            if t in pos_set:
                tpl_idx = int(torch.randint(0, len(TEMPLATES), (1,), generator=g))
                fact, query, ans = render_needle(tpl_idx, g)
                turns.append(fact)
                needles.append((fact, query, ans))
            else:
                turns.append(CHITCHAT[int(torch.randint(0, len(CHITCHAT), (1,), generator=g))])
        sessions.append(SessionData(session_id=s, turns=turns, needles=needles))
    return sessions


# -------------------------------------------------------------------------
def is_factlike(t: str) -> bool:
    return any(t.lower().startswith(prefix) for prefix in (
        "on ", "my ", "i ", "the "
    )) and any(c.isdigit() or c.isupper() for c in t)


def eval_crystal(sessions, queries, embedder, dim, n_heads):
    m = CrystalMem(dim=dim, n_heads=n_heads, embedder=embedder)
    t0 = time.perf_counter()
    for s in sessions:
        for turn in s.turns:
            if is_factlike(turn):
                m.add(turn, metadata={"session": s.session_id})
    t_ingest = time.perf_counter() - t0
    correct = 0
    for fact, query, answer in queries:
        results = m.search(query, top_k=1)
        if results and answer.lower() in results[0][0].content.lower():
            correct += 1
    return correct / len(queries), t_ingest, len(m.entries)


def eval_mem0_buffer(sessions, queries, embedder, dim, K=20):
    facts = []
    for s in sessions:
        for turn in s.turns:
            if is_factlike(turn):
                facts.append(turn)
    bank = facts[-K:]
    t0 = time.perf_counter()
    bank_emb = torch.stack([embedder(f) for f in bank]) if bank else None
    t_ingest = time.perf_counter() - t0
    if bank_emb is None:
        return 0.0, 0.0, 0
    bn = bank_emb / bank_emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    correct = 0
    for fact, query, answer in queries:
        q = embedder(query)
        qn = q / q.norm().clamp_min(1e-12)
        sims = qn.unsqueeze(0) @ bn.t()
        top = int(sims.squeeze(0).argmax().item())
        if answer.lower() in bank[top].lower():
            correct += 1
    return correct / len(queries), t_ingest, len(bank)


def eval_bank_only(sessions, queries, embedder, dim):
    facts = []
    for s in sessions:
        for turn in s.turns:
            if is_factlike(turn):
                facts.append(turn)
    if not facts:
        return 0.0, 0.0, 0
    t0 = time.perf_counter()
    bank_emb = torch.stack([embedder(f) for f in facts])
    t_ingest = time.perf_counter() - t0
    bn = bank_emb / bank_emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    correct = 0
    for fact, query, answer in queries:
        q = embedder(query)
        qn = q / q.norm().clamp_min(1e-12)
        sims = qn.unsqueeze(0) @ bn.t()
        top = int(sims.squeeze(0).argmax().item())
        if answer.lower() in facts[top].lower():
            correct += 1
    return correct / len(queries), t_ingest, len(facts)


def run(n_sessions=5, turns_per=30, needles_per=4, dim=384, seed=0):
    print("=" * 80)
    print(f"LoCoMo-style benchmark | sessions={n_sessions} × {turns_per} turns,"
          f" needles/session={needles_per}, dim={dim}")
    print("=" * 80)

    embedder = sentence_transformer_embedder(target_dim=dim)
    sessions = make_locomo_sessions(n_sessions, turns_per, needles_per, seed)
    # Cross-session questions: collect all needles from sessions[0..N-2]
    queries = []
    for s in sessions[:-1]:
        queries.extend(s.needles)
    print(f"\nTotal facts: {sum(len(s.needles) for s in sessions)}")
    print(f"Cross-session queries (asked at end): {len(queries)}")

    methods = [
        ("mem0_buffer K=20", lambda: eval_mem0_buffer(sessions, queries, embedder, dim, K=20)),
        ("bank_only ceiling", lambda: eval_bank_only(sessions, queries, embedder, dim)),
        ("crystal_h1",        lambda: eval_crystal(sessions, queries, embedder, dim, n_heads=1)),
        ("crystal_h4",        lambda: eval_crystal(sessions, queries, embedder, dim, n_heads=4)),
    ]

    print(f"\n{'method':<22} {'recall':>8} {'ingest_s':>10} {'stored':>8}")
    results = {}
    for name, fn in methods:
        acc, t_ingest, stored = fn()
        results[name] = {"recall": acc, "ingest_s": t_ingest, "stored": stored}
        print(f"{name:<22} {acc:>8.3f} {t_ingest:>10.2f} {stored:>8}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", type=int, default=5)
    p.add_argument("--turns", type=int, default=30)
    p.add_argument("--needles", type=int, default=4)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    res = run(n_sessions=args.sessions, turns_per=args.turns,
              needles_per=args.needles, dim=args.dim)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
