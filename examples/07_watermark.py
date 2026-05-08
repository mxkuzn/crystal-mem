"""07 — watermark a crystal for provenance verification.

Watermarking embeds a cryptographically-derived signature into the crystal.
The signature survives export/import/merge and can be detected by anyone
who knows the label, with FP <0.003% (see `tests/test_watermark.py`).

Use cases:
    - "This curated knowledge crystal was published by my team"
    - IP protection for shared knowledge
    - Tamper-detection (watermark vanishes if the crystal is rebuilt)
"""
from crystal_mem import CrystalMem
from crystal_mem.watermark import watermark_inject, watermark_detect


# Publisher curates a crystal of facts. Watermark detection is a statistical
# test — needs ≥20 entries for clean separation between signal and noise.
publisher = CrystalMem(dim=512, n_heads=4)
facts = [
    "Use list comprehensions over filter+map",
    "Prefer pathlib over os.path",
    "Avoid mutable default arguments",
    "Use dataclasses(frozen=True) for value objects",
    "Type hints accelerate refactoring",
    "Use match-case in Python 3.10+ for state machines",
    "Prefer | over Optional[T] (Python 3.10+)",
    "Use TypedDict for structured dicts",
    "Avoid bare except — catch specific exceptions",
    "Use f-strings, not %-formatting",
    "Use itertools.chain over sum(..., [])",
    "Use enumerate() instead of range(len())",
    "Use zip() for parallel iteration",
    "Use collections.Counter for tally",
    "Use functools.lru_cache for pure-function memoization",
    "Use async with for context-managed coroutines",
    "Use typing.Protocol for structural interfaces",
    "Use dataclass field(default_factory=list) not default=[]",
    "Use logging module not print() in libraries",
    "Use __slots__ for memory-tight value objects",
]
for fact in facts:
    publisher.add(fact, tags={"published"})

# Inject watermark — needs a label only the publisher knows
watermark_inject(publisher, label="acme-python-style-v1")

# Subscriber merges the published crystal into their memory
me = CrystalMem(dim=512, n_heads=4)
me.add("My personal preference: 4-space indent")
me.merge(publisher)

# Anyone with the label can verify provenance — even after merge
result = watermark_detect(me, label="acme-python-style-v1")
print(f"detected: {result.detected}, z-score: {result.z_score:.2f} (threshold {result.threshold})")

# Wrong label => no detection (no false positives)
fake = watermark_detect(me, label="evilcorp-stolen-data")
print(f"fake label detected: {fake.detected}, z-score: {fake.z_score:.2f}")
