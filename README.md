# CT Protocol Finder

AI-powered tool for searching, downloading, and analyzing clinical trial protocols from ClinicalTrials.gov — focused on **Clinical Pharmacology (CP)** studies.

Built as part of research on how LLM/AI agents can assist in reviewing clinical trial protocols.

## What It Does

```
Natural language query
  → LLM parses intent (study types, conditions, phases)
    → Multi-strategy search on ClinicalTrials.gov API v2
      → CP Agent scores relevance (FIH, dose escalation, PK studies prioritized)
        → LLM ranks results
          → Download protocol/SAP/ICF PDFs
            → Structured PDF extraction with docling (sections, tables)
              → LLM generates CP assessment
```

## Features

| Feature | Description |
|---------|-------------|
| **Smart Query Parsing** | LLM understands "first in human dose escalation" as study design concepts, not conditions |
| **Multi-Strategy Search** | 3 parallel search strategies with synonym expansion for maximum recall |
| **CP Agent Scoring** | Heuristic scoring (0-100) based on modality, study design, phase, documents |
| **CP Advisory** | GPT-4o-powered expert recommendations on best protocols for your research |
| **PDF Extraction** | docling-based structured extraction — preserves tables, sections, headings |
| **Structured Preview** | Protocol assessment: drug, study design, PK elements, dosing, CP relevance |
| **Evaluation Framework** | Automated metrics: query parsing accuracy, search recall, CP scoring validation |

## Architecture

- **LLM Engine**: Multi-provider (OpenAI / BullsAI / Anthropic), per-task model selection
  - `gpt-4o-mini` for parsing & ranking (fast, cheap)
  - `gpt-4o` for CP advisory (smarter reasoning)
- **ClinicalTrials.gov API v2**: Study metadata + document detection
- **PDF CDN**: Direct download via `cdn.clinicaltrials.gov/large-docs/`
- **docling**: Structured PDF → Markdown + tables + sections
- **CP Agent**: Rule-based relevance scoring + LLM advisory

## Quick Start

```bash
# Clone
git clone https://github.com/euibeom135-cell/Automation.git
cd Automation/WebSearch_Extraction

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your OpenAI API key

# Run
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000

# Open http://localhost:8000
```

## Configuration (.env)

```bash
# LLM Provider: openai | bullsai | anthropic
LLM_PROVIDER=openai

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_CP_ADVISORY_MODEL=gpt-4o

# BullsAI (UB campus, requires VPN)
BULLSAI_API_KEY=
BULLSAI_BASE_URL=https://gateway.bullsai.buffalo.edu/v1
BULLSAI_MODEL=openai/gpt-oss-120b

# Anthropic
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

## Evaluation

Built-in evaluation framework with 8 seed test queries across 3 metrics:

```bash
# Run full evaluation
python3 evaluation.py run

# Compare before/after changes
python3 evaluation.py compare

# Build ground truth (interactive labeling)
python3 evaluation.py build-ground-truth

# Evaluate CP scoring against your labels
python3 evaluation.py gt-eval
```

### Baseline Results (March 2026)

| Metric | Mean | Description |
|--------|------|-------------|
| Query Parsing | 76.0% | LLM correctly classifies study types, phases |
| Search Recall | 87.5% | Finds trials with downloadable documents |
| CP Scoring | 95.9% | Heuristic score correlates with CP relevance |
| **Overall** | **86.5%** | |

## Project Structure

```
WebSearch_Extraction/
  .env                    # API keys (not committed)
  requirements.txt        # Python dependencies
  models.py               # Pydantic data models
  ctgov_client.py         # ClinicalTrials.gov API v2 wrapper
  llm_engine.py           # Multi-provider LLM engine
  cp_agent.py             # CP relevance scoring + advisory
  study_type_synonyms.py  # Clinical trial synonym expansion
  pdf_extractor.py        # docling-based PDF extraction
  evaluation.py           # Automated evaluation framework
  app.py                  # FastAPI web server
  templates/index.html    # Single-page frontend
  static/style.css        # Styles
```

## APIs Used

- **ClinicalTrials.gov API v2** — free, no key needed
- **OpenAI API** — GPT-4o-mini (parsing/ranking), GPT-4o (CP advisory)
- **ClinicalTrials.gov CDN** — direct PDF downloads

## Roadmap

- [x] Natural language search with LLM parsing
- [x] Multi-strategy search with synonym expansion
- [x] CP Agent scoring and advisory
- [x] Structured PDF extraction (docling)
- [x] Evaluation framework
- [ ] Literature search (Semantic Scholar + OpenAlex)
- [ ] RAG-powered protocol Q&A
- [ ] Public deployment

## License

Research project — University at Buffalo.
