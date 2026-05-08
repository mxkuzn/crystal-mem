# CrystalMem — Layers above the library

**Read in:** [English](USE_CASES.md) · [Русский](USE_CASES.ru.md)


The base library is a **memory primitive**, not a product. Real product value
comes from **agentic services built on top** that exploit the algebraic
properties:

- **Mathematical merge** — federated sum without conflict resolution.
- **Algebraic watermark** — provenance unforgeable without secret label.
- **Privacy-mode retrieval** — entire bank can be discarded; only the crystal stays.
- **Compositional bind** — keys derivable algebraically from other keys.
- **Exact subtract** — math-clean forget without retraining.

Below: four product layers each elevating CrystalMem from "JSON + embedding
bank" to a service worth paying for. Each layer tags which algebraic
properties it critically depends on.

---

## Layer 1 — Cross-Tool AI Memory Sync 🌐

**Tagline:** "Your AI remembers you everywhere — ChatGPT, Claude, Cursor, Devin, locally."

**For:** Power users of multiple AI tools.

**Pain it solves:**

Today each AI app has its own memory. ChatGPT's "memory" feature doesn't
talk to Claude's projects. Cursor doesn't know what you told Claude Code.
The same context gets repeated five times a day.

**What the layer does:**

1. **Single user crystal** stored locally (`~/.crystal_mem/alice.crystal`)
   with optional E2E-encrypted sync via cloud (privacy_key per user).

2. **MCP servers everywhere** — already built. Claude Desktop / Cursor /
   any MCP-supporting client picks up the same memory.

3. **Selective merge per app**:
   - ChatGPT instance gets `filter=lambda e: "personal" not in e.tags`.
   - Cursor gets `filter=lambda e: e.metadata.get("project") == current_repo`.
   - Same source-of-truth, app-specific projection.

4. **Forget propagates everywhere**: "delete my address" → exact subtract,
   no app retains it.

5. **Migrate from Mem0 / Letta in one command** (already shipped).

**Algebraic property exploited:** filtered export (per-app projections),
exact forget, privacy-key isolation across users on same machine.

**Status:** primitives all built. Need: CLI tool, optional cloud sync server,
distribution as desktop app or homebrew package.

**Productization gap:**
- Desktop tray app (macOS / Win) that runs MCP server in background.
- Optional E2E-encrypted cloud sync (NOT raw embeddings — only crystal +
  encrypted metadata).
- Browser extension for ChatGPT web.

---

## Layer 2 — Enterprise Compliance Memory Bus 🏛️

**Tagline:** "Single audit-grade memory layer for every LLM in your company."

**For:** Banks, healthcare, EU enterprises post-AI Act, regulated SaaS.

**Pain it solves:**

GDPR Article 17 mandates exact erasure. Most companies fake this with
soft-delete or filter. When the regulator comes — fines.

Today each team's LLM tool has its own memory store. No central audit.
No watermarking on AI outputs. Cannot prove "this output came from our
model on date X".

**What the layer does:**

1. **Centralized memory bus**: every LLM-using service in the org talks
   through CrystalMem-as-a-service.

2. **Watermark every AI-generated output**: invisible signature in the
   crystal so provenance is provable later.

3. **Cryptographic-clean forget on customer request**: ~50 µs/entry,
   guaranteed math-exact (Δ ≈ 4×10⁻⁶). Audit log records each forget.

4. **Federated across departments**: HR, Sales, Legal each have their
   own crystal. Combined view via merge for executives, but raw data
   stays in department.

5. **Selective disclosure to regulators**: "show all decisions about
   loan applications" → filtered export. No leak of unrelated data.

6. **DPO dashboard**: capacity utilization, forget audit log, watermark
   verifications, compliance reports.

**Algebraic property exploited:** all five.
- Federated merge for cross-department.
- Watermark for output provenance.
- Exact subtract for GDPR.
- Privacy-mode for customer-grade isolation.
- Compositional bind for "all decisions about X".

**Status:** primitives shipped. Need: enterprise-grade infra
(K8s deployment, RBAC, SSO, audit DB).

**Productization gap:**
- B2B SaaS with $50K-500K ACV.
- SOC2 / ISO27001 certifications.
- Reference customers in regulated industries.

---

## Layer 3 — Memory Marketplace ("GitHub for AI Memory") 🏪

**Tagline:** "Curated knowledge as a portable artifact. Subscribe, fork, merge."

**For:** Developers, professionals, hobbyists wanting to elevate their AI's expertise.

**Pain it solves:**

Today users stuff "Python idioms 2026", "current GDPR rules", "OWASP top-10
latest", or "your company's coding conventions" into a system prompt that
costs tokens forever and can't be versioned.

**What the layer does:**

1. **Public crystal hub** like npm/pip but for memory:
   - `crystal install python_idioms_2026`
   - `crystal install gdpr_compliance_eu_2026`
   - `crystal install acme_corp_conventions`

2. **Versioned crystals**: maintainers ship updates as `unmerge(v1) +
   merge(v2)` patches.

3. **Verified provenance via watermark**: "this Python idioms crystal
   was published by Real Python team" — z-score check.

4. **Forking and merging**: clone, modify, push back. PRs as new crystals
   that maintainers can merge.

5. **Filtered subscription**: "I only want safety-related rules from
   OWASP, not the full set" → filtered import.

**Algebraic property exploited:**
- Federated merge for combining crystals.
- Watermark for trust.
- Selective filter for partial subscription.
- Unmerge for version revocation.

**Status:** `crystal_mem/demos/d6_marketplace.py` proves the mechanics. Need: hub site,
discovery, payment for premium crystals.

**Productization gap:**
- Web hub (like npm.org).
- Search / discovery UX.
- Payment for premium crystals (medical guidelines, legal precedents).
- Stripe + revenue share with publishers.

**Network effect risk:** chicken-and-egg — without crystals nobody comes;
without users nobody publishes. Bootstrap with ~50 community crystals first.

---

## Layer 4 — Multi-Agent Coordination Workspace 🤖

**Tagline:** "Multiple AI agents share a workspace. Each contributes, each can revoke."

**For:** Multi-agent systems (LangGraph, AutoGen, CrewAI, swarms).

**Pain it solves:**

When 5 agents collaborate, each has its own context. Sharing information
requires explicit message-passing. Conflicts when two agents edit the same
fact. No way to "undo agent X's contribution" cleanly.

**What the layer does:**

1. **Shared workspace crystal** that all agents in a swarm read/write.

2. **Per-agent provenance**: every entry stamped with which agent added it.

3. **Conflict-free updates**: HRR sum is commutative — order of writes
   doesn't matter. No CRDT vendor lock-in.

4. **Surgical revoke**: agent X malfunctioned and added 50 bad facts —
   `unmerge(agent_x_export)` removes only its contributions.

5. **Lineage replay**: see chronological order of agent contributions for debugging.

6. **Federated swarms**: 3 separate agent swarms can merge their workspaces
   into a meta-workspace; later split back if collaboration ends.

**Algebraic property exploited:**
- Mathematical merge (no conflict resolution needed).
- Exact subtract for revoke.
- Provenance for debugging.
- Federated merge for swarm-of-swarms.

**Status:** primitives ready, no integration with major multi-agent frameworks yet.

**Productization gap:**
- LangGraph / AutoGen / CrewAI plugins.
- Multi-tenant workspace server.
- Real-time collaborative editing visualization.

---

## Other vertical applications worth listing

| # | Vertical | Hook | Critical algebraic property |
|---|---|---|---|
| 5 | Game NPC continuity | NPCs remember player decisions, federated across save files | merge / unmerge |
| 6 | Therapy AI (HIPAA) | Patient sessions with crypto-clean forget on demand | exact subtract |
| 7 | Adversarial / red-team logbook | Pen-test findings with signed provenance | watermark |
| 8 | Edge IoT swarm | Sensors aggregate observations into 4KB crystal | privacy-mode retrieval |
| 9 | RLHF data manager | Forget bad labels exactly without retrain | exact subtract |
| 10 | AI persona continuity (Replika-likes) | Character memory portable across platforms | crystal export |
| 11 | Code review memory | Per-team conventions as a published crystal | marketplace + filter |
| 12 | Climate model ensemble | N forecasts merged without sharing model weights | federated merge |
| 13 | DSL / grammar memory | Code-completion patterns retrieved compositionally | compositional bind |
| 14 | Smart contract state aggregation | DAO votes as crystals merged on-chain | mathematical merge |
| 15 | Cross-language knowledge transfer | EN crystal merged with JA crystal | federated merge |
| 16 | Personality fingerprint detection | Writing style as crystal, watermarks detect impostors | watermark |
| 17 | AGI alignment safeguard | "Forbidden actions" crystal checked before each agent decision | compositional bind |
| 18 | Auto-curriculum learning | Track recall decay, prompt review when capacity drops | capacity awareness |
| 19 | A/B prompt evaluator | Each variant's user feedback as a crystal, compare via diff | diff + filter |

---

## Bottom line

The library by itself is **infrastructure**. Like Postgres alone is "store
rows + query". The value comes from what gets built **on top**:

- Postgres → Supabase / Neon / Heroku Postgres
- Redis → Upstash / Vercel KV
- Vector DB → Pinecone / Weaviate
- **CrystalMem → Memory Sync / Compliance Bus / Marketplace / Multi-Agent**

Each layer above represents a 10×-grade product over what is achievable with
JSON+FAISS. Pick one and build it as a service.
