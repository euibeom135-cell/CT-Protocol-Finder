# CLAUDE.md — Clinical_Trials_Finder

This file is read automatically by Claude Code at the start of every session in this project. It captures architecture, design decisions, and context that isn't obvious from the code alone.

User grounding (preferences, working style, research context) lives in Claude Code memory at `~/.claude/projects/-Users-euibeoms-Desktop-Automation/memory/` — that's loaded automatically too.

---

## 📬 Notice for Claude cowork (if you're reading this from the mounted share folder)

If you see this file at `/Users/euibeoms/Clinical_Trials_Finder_cowork/CLAUDE.md` (or similar share path) rather than `/Users/euibeoms/Desktop/Automation/Clinical_Trials_Finder/CLAUDE.md`, you're reading the **auto-synced read-only mirror**. Important:

- **Do not write to this folder.** A background rsync with `--delete-excluded` runs after every Claude Code edit; any file you create here will be deleted on the next sync.
- **The real project folder is** `/Users/euibeoms/Desktop/Automation/Clinical_Trials_Finder/` — only Claude Code writes there.
- **To propose an edit or new file:** produce a markdown block with the target file path and full new content. Euibeom will paste it to Claude Code, which will apply it to the real folder. The change will then appear in this mirror within ~1 second.
- **Excluded from the mirror:** `downloads/` (80 MB of PDFs), `data/` (FAISS index), `__pycache__/`, `.claude/`, `static/`, `templates/`, `tests/`, `render.yaml`, `.env`. If you need something that's excluded, ask Euibeom.

---

---

## What this project is

A FastAPI web app for searching, downloading, and analyzing clinical trial protocols from ClinicalTrials.gov, with a focus on **Clinical Pharmacology (CP)** studies — especially **First-in-Human (FIH)** and dose-escalation trials. Built as part of research on how LLM/AI agents can assist in reviewing clinical trial protocols.

Former name: `WebSearch_Extraction` (renamed 2026-04-11, commit `0dc7986`). The README internally refers to it as "CT Protocol Finder."

## Research context (important)

This project is part of a larger research program at Euibeom's clinical pharmacology lab at UB:

- **Current paper (in progress):** ChatGPT evaluation on extracting PK/PD/DDI from FDA prescribing information — the "Table 2 DDI grading framework" work.

- **Funder context (Gates Foundation / Shawn Dolley).** The real driving use case is Gates Foundation's internal clinical protocol review bottleneck. Shawn Dolley's ask (paraphrased): *"If you can review a protocol 10x faster than we do here, I'll give you whatever money you need."* The Gates team currently has a human Clinical Pharmacology reviewer as the bottleneck when deciding whether to fund incoming clinical protocols. **The FIH Comparator Finder is a subcomponent of a larger protocol-review tool, not a standalone project.** Shawn is a management consultant (digital transformation / data strategy / OHDSI), **NOT a clinical pharmacologist** — he is the funder, not the scientific validator, and should not be treated as one. UB was funded specifically because of the PKPD + LLM combination. Euibeom's existing `/CP` and `/cp-review` skills (outside this repo) are the seed of Shawn's ask.

- **Next-chapter initiative:** The **FIH Comparator Finder** — an agentic AI that takes a new FIH drug candidate, finds comparator compounds, pulls their FDA Clinical Pharmacology reviews, extracts FIH design parameters, and synthesizes recommendations (scoped per the open questions below — not necessarily a concrete starting-dose number). **Keeps the existing trial-design similarity engine (it's useful for a different question) but adds a NEW drug-pharmacology similarity layer as the actual Phase 1 engine — see the "Status of the multi-axis similarity question" resolution section at the bottom of this file.** Full roadmap in `~/.claude/projects/-Users-euibeoms-Desktop-Automation/memory/project_fih_comparator.md`.

- **Test molecule for first vertical slice: bedaquiline.** Chosen because (a) Gates-aligned (TB), (b) novel MoA — stress-tests the "no obvious comparator" edge case, (c) FDA CP review is publicly available. Cowork session 2 will run the existing similarity engine against bedaquiline and use the real output + direct code read of `ontology_engine.py` / `ontology_data/fih_taxonomy.py` to answer the multi-axis similarity design question below with evidence rather than opinion.

When Euibeom refers to "the FIH agent," "comparator finder," or "extending this for FIH work," he means this initiative.

### Open questions (from cowork session 3, 2026-04-11 — do not assume answers)

1. **Is "DAC" = TB Drug Accelerator specifically, or broader Gates consortium?** Affects scope of the Gates use case and which therapeutic areas the tool should optimize for.
2. **First user of the tool = Gates internal CP reviewer, senior CP consultant, or Gates-funded partner's small CP team?** Affects output format, how much domain knowledge to assume, and how aggressive the synthesis layer can be.
3. **Does the agent stop at "evidence package + comparators" or go all the way to a concrete starting-dose recommendation?** The latter crosses into liability territory tied to the "confident wrong answer" failure mode from the current paper. This is the single biggest scoping decision.
4. **Scope = FIH-only, or "first-in-*this*-population"?** Same technical machinery, but reframing as "first-in-THIS-population" (pediatric, pregnant, malnourished, co-infected) has a much bigger ceiling for Gates because many Gates molecules go into populations the original FDA FIH never studied. Flagged by cowork session 3 as a framing to explore, not a commitment.

## Repo layout (critical nuance)

```
/Users/euibeoms/Desktop/Automation/        ← git repo root (NOT inside the project folder)
├── .git/                                    ← git lives here, not in Clinical_Trials_Finder/
├── Clinical_Trials_Finder/                  ← this project (flat Python layout)
│   ├── app.py                               ← FastAPI server entry point
│   ├── cp_agent.py                          ← CP relevance scoring (FIH-weighted)
│   ├── ctgov_client.py                      ← ClinicalTrials.gov API v2 wrapper
│   ├── llm_engine.py                        ← Multi-provider LLM (OpenAI/BullsAI/Anthropic)
│   ├── pdf_extractor.py                     ← docling-based PDF → structured sections
│   ├── fda_client.py                        ← openFDA Drug Label + Drugs@FDA client
│   ├── similarity_engine.py                 ← FAISS + embeddings + ontology (entry: find_similar_by_nct, find_similar_by_text)
│   ├── ontology_engine.py                   ← 4-layer composite similarity
│   ├── study_type_synonyms.py               ← FIH/PK/DDI synonym expansion
│   ├── build_trial_index.py                 ← Builds the FAISS index from CT.gov trials
│   ├── evaluation.py                        ← Automated eval framework
│   ├── models.py                            ← Pydantic data models
│   ├── ontology_data/
│   │   ├── fih_taxonomy.py                  ← 7-dimension FIH classification (see below)
│   │   ├── drug_classes.py                  ← Drug class / mechanism mappings
│   │   ├── mesh_subset.py                   ← MeSH condition subset
│   │   └── __init__.py
│   ├── templates/index.html                 ← Single-page frontend
│   ├── static/style.css
│   ├── data/                                ← FAISS index files (gitignored)
│   ├── downloads/                           ← Downloaded PDFs (gitignored)
│   ├── .env                                 ← API keys (gitignored)
│   ├── .env.example
│   ├── requirements.txt
│   ├── README.md                            ← stale in places (see "Known stale docs" below)
│   └── CLAUDE.md                            ← this file
└── (other subfolders may exist for other projects in this repo)
```

**⚠️ Git commands must be run from `/Users/euibeoms/Desktop/Automation/`, not from inside `Clinical_Trials_Finder/`.** Using `git status` from the project folder will still work (git walks up to find `.git/`), but staged paths will be relative to the repo root, i.e. `Clinical_Trials_Finder/app.py` not `app.py`.

## Stack

- **Backend:** FastAPI + Uvicorn (not Flask)
- **Frontend:** Jinja2 templates + vanilla static CSS, single-page
- **LLM:** Multi-provider engine — `gpt-4o-mini` for parsing/ranking, `gpt-4o` for CP advisory. Providers: OpenAI, Anthropic, BullsAI (UB campus gateway, requires VPN)
- **Data sources:** ClinicalTrials.gov API v2 (free, no key), ClinicalTrials.gov CDN for PDFs, openFDA Drug Label + Drugs@FDA APIs, FDA accessdata for CP review PDFs
- **PDF parsing:** `docling` (structured extraction, preserves tables/sections)
- **Similarity:** `faiss-cpu` + OpenAI `text-embedding-3-small` (1536 dims)

## FIH taxonomy — the project's core IP

The file `ontology_data/fih_taxonomy.py` defines a 7-dimension classification vocabulary for FIH/dose-escalation trials. The docstring calls this a **novel contribution** — no published ontology captures FIH-specific design elements at this level of granularity. The dimensions and their weights for similarity scoring:

| Dimension | Weight | Example tags |
|---|---|---|
| `pk_design` | 25% | SAD, MAD, food effect, DDI, bioavailability, mass balance/ADME, QTc, renal/hepatic impairment |
| `modality` | 20% | small molecule, mAb, ADC, bispecific, CAR-T, siRNA, mRNA, peptide, gene therapy, PROTAC |
| `escalation_method` | 15% | 3+3, mTPI, mTPI-2, BOIN, CRM, EWOC, accelerated titration, rolling-six, i3+3 |
| `population` | 15% | healthy volunteers, advanced/metastatic patients, Japanese bridging, pediatric, elderly, organ-impaired |
| `route` | 10% | IV, SC, oral, IM, intrathecal, inhaled, topical, ophthalmic |
| `dlt_window` | 8% | 21-day, 28-day, 42-day, cycle 1 |
| `pd_biomarkers` | 7% | receptor occupancy, target engagement, cytokine panel, PK/PD modeling, immunogenicity |

Extraction is regex-based: `extract_fih_profile(text)` returns an `FIHProfile` dataclass. Similarity between two profiles uses weighted Jaccard per dimension (`fih_profile_similarity()`).

## Composite similarity (ontology_engine.py)

> **⚠️ This is a TRIAL-DESIGN similarity engine, not a drug-pharmacology similarity engine.** It answers "what FIH trials have similar *designs*" by extracting design features (escalation method, population type, route, etc.) from trial titles and brief summaries. It does NOT answer "what drugs are pharmacologically similar" — the axes a CP reviewer needs for FIH comparator selection (tox profile, ADME, MoA, DDI, dose-selection basis) are not in the taxonomy and are not extractable from trial metadata anyway. The new drug-pharmacology similarity layer will be built as separate Phase 1 work; **don't try to squeeze drug-pharmacology axes into this engine.** See the "Status of the multi-axis similarity question" resolution section at the bottom of this file.

The trial-design similarity search combines four signals:

- **25%** MeSH condition distance (via `ontology_data/mesh_subset.py`)
- **30%** Drug class / mechanism similarity (via `ontology_data/drug_classes.py`)
- **30%** FIH taxonomy similarity (via `ontology_data/fih_taxonomy.py`)
- **15%** Text embedding cosine similarity (FAISS + OpenAI embeddings)

Entry points in `similarity_engine.py`:
- `find_similar_by_nct(nct_id)` — find trials similar to a known NCT ID
- `find_similar_by_text(text)` — find trials similar to arbitrary text (e.g. an uploaded candidate drug description)

A pre-built FAISS index is required. Build it with `python3 build_trial_index.py`. The index files live in `data/` (gitignored).

## CP relevance scoring (cp_agent.py)

`compute_cp_relevance(study)` returns a 0–100 `cp_score` based on:
- **Design keywords (50%)** — FIH designs get the highest weights (FIH/SAD/MAD = 10, dose escalation = 9, food effect/mass balance = 9, PK = 9)
- **Modality (30%)** — small molecule/oral/IV/SC score highest (≥0.9); topicals, vaccines, devices score near zero
- **Phase bonus (10%)** — Phase 1 gets +2, Phase 2 gets +1, Phase 3/4 get nothing
- **Document bonus (10%)** — having both Protocol + SAP available = +2
- **Enrollment factor** — 10–200 subjects is "sweet spot" for early-phase PK, >1000 is penalized

Red flags (non-CP interventions: vaccine, ointment, device, behavioral, etc.) can force a study below the relevance threshold. `filter_for_cp(studies)` returns `(study, cp_info)` tuples sorted by score, filtered to CP-relevant.

## Running locally

```bash
cd /Users/euibeoms/Desktop/Automation/Clinical_Trials_Finder
pip install -r requirements.txt
cp .env.example .env  # then edit to add OPENAI_API_KEY
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

For similarity search to work, also run: `python3 build_trial_index.py` (builds the FAISS index into `data/`).

## Config (.env)

Required:
- `OPENAI_API_KEY` — for LLM parsing/ranking/advisory AND for embeddings used by `similarity_engine.py`
- `OPENAI_MODEL=gpt-4o-mini` (parsing/ranking)
- `OPENAI_CP_ADVISORY_MODEL=gpt-4o` (CP advisory)

Optional:
- `BULLSAI_API_KEY`, `BULLSAI_BASE_URL`, `BULLSAI_MODEL` (UB campus gateway)
- `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`

## Known stale docs

`README.md` is partially outdated:
- Quick Start still says `cd Automation/WebSearch_Extraction` (should be `Clinical_Trials_Finder`)
- Project Structure section doesn't mention `similarity_engine.py`, `ontology_engine.py`, `ontology_data/`, `fda_client.py`, or `build_trial_index.py` — all of which exist and matter
- Roadmap still has unchecked items ("Literature search", "RAG-powered protocol Q&A") that may reflect current state, but the checked items haven't been updated for recent additions

A README refresh is on the TODO list but hasn't been done yet. Don't treat the README as authoritative — this CLAUDE.md is newer.

## Status of the multi-axis similarity question (resolved by cowork session 3, 2026-04-11)

This section previously captured an unresolved design concern that the current engine "collapses similarity into a single weighted composite." **Cowork session 3 resolved it with direct code reads, not opinion.** Three findings, from least to most consequential:

### Finding 1 — Top-level axes: already exposed (no action needed)

`SimilarityBreakdown` in `ontology_engine.py` (lines 82–104) already stores `mesh_score`, `drug_class_score`, `fih_score`, and `embedding_score` as **separate fields**. `rerank_results()` attaches them to each result via `similarity_breakdown`. The composite score is only used as the sort key. The four top-level axes do not need to be un-collapsed — they are already separate in the data model.

### Finding 2 — FIH sub-dimensions: silently collapsed (tiny fix available)

`fih_dimension_breakdown()` exists in `fih_taxonomy.py` (line 274) and can return per-dimension similarity for the 7 FIH axes (`pk_design`, `modality`, `escalation_method`, `population`, `route`, `dlt_window`, `pd_biomarkers`). But `ontology_engine.py` never calls it — it only calls `fih_profile_similarity()`, which pre-averages the 7 dimensions. The sub-scores are computed and then discarded.

**Tiny fix (~5 lines, opportunistic side-quest):** add `fih_sub_breakdown: dict[str, float]` to `SimilarityBreakdown`, populate from `fih_dimension_breakdown()` inside `compute_similarity()`, surface in `to_dict()`. Low risk, no behavior change to existing callers. Worth doing independently of the larger drug-pharmacology work.

### Finding 3 — The current engine answers the wrong question for the FIH Comparator use case

This is the one that changes project scope.

The 7 FIH taxonomy dimensions measure **how the trial was designed** (escalation method, DLT window, population type, etc.), not **what the drug is pharmacologically**. None of the axes a CP reviewer actually needs for FIH comparator selection — tox profile, ADME, target/MoA, DDI inhibition/induction, dose-selection basis (NOAEL vs MABEL) — are in the taxonomy. And none of them are extractable from trial titles and brief summaries anyway (which is all the regex extractor sees).

**The current engine answers "what trials have similar designs" — a useful question, but not "what drugs are pharmacologically similar enough that their FDA CP review would inform my new molecule's FIH."** Those are different questions with different data sources.

### Consequences (the plan going forward)

1. **Keep the existing engine. Rename/document it for honesty.** Frame `ontology_engine.py` + `similarity_engine.py` as a **"trial-design similarity engine"** — it's useful for "find me other FIH trials that look like this one." Don't delete. Don't conflate with drug-pharmacology similarity. The framing warning has been added at the top of the "Composite similarity" section above.

2. **Build a NEW drug-pharmacology similarity layer as Phase 1 work.** This is the actual FIH Comparator Finder engine. Inputs: **FDA Clinical Pharmacology review PDFs**, not trial metadata. Axes: tox profile, ADME, target/MoA, DDI profile, dose-selection basis. Extraction: reuses `pdf_extractor.py` + docling. **This is new code, not an extension of `ontology_engine.py`.** Do NOT start implementing — cowork will produce a spec first, derived from a by-hand walk-through of the bedaquiline FDA CP review.

3. **Division of labor (confirmed cowork session 3):** cowork = planning agent (specs, design critiques, research context, funder alignment). Claude Code = implementation (code, git, tests). Cowork does not write to the mirror folder; Claude Code does not run planning or research-context synthesis without a spec.

### Next concrete steps

- **Cowork's next session:** Euibeom and cowork will walk through the bedaquiline FDA CP review by hand to define what the new drug-pharmacology similarity layer needs to extract. That by-hand pass becomes the spec Claude Code implements against. Until that spec lands, Claude Code should not start implementing the drug-pharmacology layer.
- **Claude Code's optional side quest (safe to do independently):** the 5-line `fih_sub_breakdown` fix from Finding 2. Ask Euibeom before implementing.

## Cross-agent coordination

Euibeom uses two Claude environments:
1. **Claude Code** (this one) — local codebase work, git, file edits
2. **Claude cowork** (Claude.ai workspace) — planning, research synthesis, literature review. Has access to ClinicalTrials.gov MCP tools and PubMed MCP tools. Has its own memory system at `~/Library/Application Support/Claude/local-agent-mode-sessions/.../memory/`

The two agents coordinate through filesystem artifacts: this `CLAUDE.md`, git history, and exportable memory files. There is no live session bridge. If you (Claude Code) make a significant architectural decision, consider writing it here so the next cowork planning session can see it; conversely, when the cowork makes a planning decision, it may land here via Euibeom.
