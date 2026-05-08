# CrystalMem — Надстройки поверх библиотеки

**Язык:** [English](USE_CASES.md) · [Русский](USE_CASES.ru.md)


База — это **memory-примитив**, не продукт. Реальная продуктовая ценность
появляется в **agentic-сервисах поверх**, использующих алгебраические
свойства:

- **Математический merge** — federated-сумма без conflict resolution.
- **Алгебраический watermark** — provenance, неподделываемый без секретного label.
- **Privacy-mode retrieval** — весь bank можно сбросить; остаётся только кристалл.
- **Compositional bind** — ключи алгебраически выводятся из других ключей.
- **Точное вычитание** — math-clean forget без переобучения.

Ниже — четыре продуктовых надстройки, каждая поднимает CrystalMem из
"JSON + embedding bank" в сервис, за который платят. Каждая помечена
тем, какие алгебраические свойства критически использует.

---

## Layer 1 — Cross-Tool AI Memory Sync 🌐

**Tagline:** «Твой AI помнит тебя везде — ChatGPT, Claude, Cursor, Devin, локально.»

**Для:** power-юзеров нескольких AI-инструментов.

**Боль:**

Сегодня каждое AI-приложение со своей памятью. ChatGPT "memory" не общается
с Claude projects. Cursor не знает что ты сказал Claude Code. Один и тот же
контекст повторяется по 5 раз в день.

**Что делает надстройка:**

1. **Единый user-кристалл** локально (`~/.crystal_mem/alice.crystal`) с
   опциональным E2E-encrypted sync через cloud (privacy_key per user).

2. **MCP-серверы везде** — уже построено. Claude Desktop / Cursor /
   любой MCP-совместимый клиент видит ту же память.

3. **Селективный merge per app**:
   - ChatGPT-инстанс получает `filter=lambda e: "personal" not in e.tags`.
   - Cursor получает `filter=lambda e: e.metadata.get("project") == current_repo`.
   - Один source-of-truth, app-специфичные projection'ы.

4. **Forget пропагируется везде**: "удали мой адрес" → точное вычитание,
   ни одно приложение не сохранило.

5. **Миграция с Mem0 / Letta одной командой** (уже отгружено).

**Какие алгебраические свойства использует:** filtered export (projection per
app), точное удаление, privacy-key изоляция между юзерами на одной машине.

**Статус:** примитивы построены. Нужно: CLI-тула, опциональный cloud-sync
сервер, дистрибуция как desktop app или homebrew package.

**Гэп до продукта:**
- Desktop tray app (macOS / Win) запускающий MCP-сервер в фоне.
- Опциональный E2E-encrypted cloud sync (НЕ raw-эмбеддинги — только
  кристалл + encrypted metadata).
- Browser-расширение для ChatGPT web.

---

## Layer 2 — Enterprise Compliance Memory Bus 🏛️

**Tagline:** «Единый audit-grade memory-слой для каждого LLM в твоей компании.»

**Для:** Банков, healthcare, EU enterprises post-AI Act, regulated SaaS.

**Боль:**

GDPR Article 17 требует точного удаления. Большинство компаний фейкуют
soft-delete или фильтром. Приходит регулятор → штрафы.

Сегодня у каждого LLM-инструмента в команде свой memory-store. Никакого
центрального аудита. Никакого watermarking на AI-выходах. Не докажешь "это
выход вышел из нашей модели на дату X".

**Что делает надстройка:**

1. **Централизованный memory-bus**: каждый LLM-сервис в org говорит через
   CrystalMem-as-a-service.

2. **Watermark на каждом AI-выходе**: невидимая подпись в кристалле,
   provenance проверяется потом.

3. **Криптографически точное удаление по запросу клиента**: ~50 µs/entry,
   гарантировано math-exact (Δ ≈ 4×10⁻⁶). Audit log записывает каждый forget.

4. **Federated по департаментам**: HR, Sales, Legal — каждый со своим
   кристаллом. Combined view через merge для исполнителей, raw-данные
   остаются в департаменте.

5. **Selective disclosure для регуляторов**: "покажи все решения по
   loan-заявкам" → filtered export. Без leaks unrelated данных.

6. **DPO dashboard**: capacity utilization, forget audit log, watermark
   verifications, compliance-отчёты.

**Какие алгебраические свойства использует:** все пять.
- Federated merge для cross-department.
- Watermark для output provenance.
- Точное вычитание для GDPR.
- Privacy-mode для customer-grade изоляции.
- Compositional bind для "all decisions about X".

**Статус:** примитивы отгружены. Нужно: enterprise-grade infra
(K8s deployment, RBAC, SSO, audit DB).

**Гэп до продукта:**
- B2B SaaS с $50K-500K ACV.
- SOC2 / ISO27001 сертификации.
- Reference customers в regulated industries.

---

## Layer 3 — Memory Marketplace («GitHub for AI Memory») 🏪

**Tagline:** «Curated knowledge как переносимый артефакт. Subscribe, fork, merge.»

**Для:** разработчиков, профессионалов, хоббистов, желающих поднять экспертизу AI.

**Боль:**

Сегодня "Python idioms 2026", "current GDPR rules", "OWASP top-10 latest"
или "company coding conventions" пихаются в system prompt — токены жгут
вечно, версионирования нет.

**Что делает надстройка:**

1. **Public crystal hub** как npm/pip но для памяти:
   - `crystal install python_idioms_2026`
   - `crystal install gdpr_compliance_eu_2026`
   - `crystal install acme_corp_conventions`

2. **Versioned crystals**: мейнтейнеры выпускают апдейты как `unmerge(v1) +
   merge(v2)` патчи.

3. **Verified provenance через watermark**: "этот Python idioms кристалл
   опубликован Real Python командой" — z-score check.

4. **Forking и merging**: клонируешь, модифицируешь, пушишь обратно. PR'ы
   как новые кристаллы которые мейнтейнеры мерджат.

5. **Filtered subscription**: "хочу только safety-rules из OWASP, не весь
   set" → filtered import.

**Какие алгебраические свойства использует:**
- Federated merge для комбинирования кристаллов.
- Watermark для trust.
- Selective filter для partial subscription.
- Unmerge для version revocation.

**Статус:** `crystal_mem/demos/d6_marketplace.py` проверяет механику. Нужно: hub-сайт,
discovery, payments для premium crystals.

**Гэп до продукта:**
- Web hub (типа npm.org).
- Search / discovery UX.
- Payments за premium crystals (медицинские гайдлайны, legal precedents).
- Stripe + revenue share с publishers.

**Network-effect risk:** chicken-and-egg — без кристаллов никто не приходит,
без юзеров никто не публикует. Bootstrap-ить ~50 community-кристаллами сначала.

---

## Layer 4 — Multi-Agent Coordination Workspace 🤖

**Tagline:** «Несколько AI-агентов разделяют workspace. Каждый контрибьютит, каждый может отозвать.»

**Для:** multi-agent systems (LangGraph, AutoGen, CrewAI, swarms).

**Боль:**

Сегодня когда 5 агентов коллаборируют — у каждого свой контекст. Sharing
информации требует явного message-passing. Конфликты когда два агента
правят один факт. Никак не "undo вклад агента X" чисто.

**Что делает надстройка:**

1. **Shared workspace crystal** который читают/пишут все агенты swarm.

2. **Per-agent provenance**: каждая запись помечена каким агентом добавлена.

3. **Conflict-free updates**: HRR-сумма коммутативна — порядок записей не
   важен. Никакого CRDT vendor lock-in.

4. **Surgical revoke**: агент X сломался и добавил 50 плохих фактов —
   `unmerge(agent_x_export)` удаляет только его вклады.

5. **Lineage replay**: chronological-порядок agent-контрибьюшинов для
   дебага.

6. **Federated swarms**: 3 отдельных swarm агентов могут смерджить
   workspaces в meta-workspace; позже расщепить когда коллаборация
   закончилась.

**Какие алгебраические свойства использует:**
- Математический merge (без conflict resolution).
- Точное вычитание для revoke.
- Provenance для дебага.
- Federated merge для swarm-of-swarms.

**Статус:** примитивы готовы, интеграции с major multi-agent фреймворками
ещё нет.

**Гэп до продукта:**
- LangGraph / AutoGen / CrewAI плагины.
- Multi-tenant workspace сервер.
- Real-time collaborative editing визуализация.

---

## Другие vertical-применения

| # | Vertical | Hook | Critical algebraic property |
|---|---|---|---|
| 5 | Game NPC continuity | NPC помнят решения игрока, federated между save-файлами | merge / unmerge |
| 6 | Therapy AI (HIPAA) | Сессии пациентов с crypto-clean forget по запросу | точное вычитание |
| 7 | Adversarial / red-team logbook | Pen-test findings с signed provenance | watermark |
| 8 | Edge IoT swarm | Сенсоры агрегируют наблюдения в 4KB-кристалл | privacy-mode retrieval |
| 9 | RLHF data manager | Forget bad labels точно без retrain | точное вычитание |
| 10 | AI persona continuity (Replika-likes) | Character-память переносимая между платформами | crystal export |
| 11 | Code review memory | Per-team conventions как опубликованный кристалл | marketplace + filter |
| 12 | Climate model ensemble | N forecasts merged без шаринга весов модели | federated merge |
| 13 | DSL / grammar memory | Code-completion patterns retrieved compositionally | compositional bind |
| 14 | Smart contract state aggregation | DAO голоса как кристаллы merged on-chain | mathematical merge |
| 15 | Cross-language knowledge transfer | EN-кристалл merged с JA-кристаллом | federated merge |
| 16 | Personality fingerprint detection | Writing style как кристалл, watermark детектит самозванца | watermark |
| 17 | AGI alignment safeguard | "Forbidden actions" кристалл, проверяемый перед каждым решением | compositional bind |
| 18 | Auto-curriculum learning | Tracking recall decay, prompt review когда capacity падает | capacity awareness |
| 19 | A/B prompt evaluator | Каждый вариант прокачан user feedback'ом как кристалл, сравнение через diff | diff + filter |

---

## Bottom line

Библиотека сама по себе — **инфраструктура**. Как Postgres сам по себе —
"хранить строки + queries". Ценность в том что строится **поверх**:

- Postgres → Supabase / Neon / Heroku Postgres
- Redis → Upstash / Vercel KV
- Vector DB → Pinecone / Weaviate
- **CrystalMem → Memory Sync / Compliance Bus / Marketplace / Multi-Agent**

Каждый layer выше — это 10×-уровень над "JSON+FAISS". Возьми один и сделай
как сервис.
