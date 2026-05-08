"""02 — plug in a real semantic embedder.

The default hash embedder gives no semantic understanding (paraphrases won't
match). For real RAG-style search, use sentence-transformers (bundled in
the [embeddings] extra).

    pip install crystal-mem[embeddings]
    python examples/02_real_embedder.py
"""
from crystal_mem import CrystalMem
from crystal_mem.embedders import sentence_transformer_embedder


# all-MiniLM-L6-v2 is 384-d, fast on CPU, ~80MB download.
# For higher quality and dim=1024 use qwen3_embedder() instead.
embedder = sentence_transformer_embedder(target_dim=384)

m = CrystalMem(dim=384, n_heads=4, embedder=embedder)

m.add("Python is dynamically typed and uses duck typing")
m.add("Cooking pasta requires plenty of salted water")
m.add("Type hints help with refactoring large Python codebases")

# Notice that "language with type annotations" matches Python entries
# even though those words aren't in the stored facts:
print("\nQuery: 'language with type annotations'")
for entry, score in m.search("language with type annotations", top_k=2):
    print(f"  [{score:+.3f}] {entry.content}")

print("\nQuery: 'how to boil water for pasta'")
for entry, score in m.search("how to boil water for pasta", top_k=2):
    print(f"  [{score:+.3f}] {entry.content}")
