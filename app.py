"""FastAPI web server for Clinical Trial Protocol Finder."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fastapi import UploadFile, File

from cp_agent import filter_for_cp, CP_ADVISOR_PROMPT
from fda_client import FDAClient
from pdf_extractor import extract_protocol, format_sections_for_llm
from ctgov_client import CTGovClient
from llm_engine import LLMEngine
from models import SearchIntent, SearchResponse
from similarity_engine import TrialSimilarityEngine
from study_type_synonyms import expand_study_types, score_study_type_match

load_dotenv()

app = FastAPI(title="CT Protocol Finder")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Shared instances
ctgov = CTGovClient()
fda = FDAClient()
llm: LLMEngine | None = None
sim_engine = TrialSimilarityEngine()

# Ensure required directories exist (needed for fresh deploys)
(BASE_DIR / "downloads").mkdir(exist_ok=True)
(BASE_DIR / "data").mkdir(exist_ok=True)

# Try to load pre-built similarity index
_data_dir = BASE_DIR / "data"
if _data_dir.exists():
    if sim_engine.load(str(_data_dir)):
        print(f"[Similarity] Loaded index with {sim_engine.total_indexed} trials")
    else:
        print("[Similarity] No pre-built index found. Run: python3 build_trial_index.py")
else:
    print("[Similarity] No data/ directory. Run: python3 build_trial_index.py")


def get_llm() -> LLMEngine:
    global llm
    if llm is None:
        llm = LLMEngine()
    return llm


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/parse")
async def parse_query(payload: dict):
    """Step 1: Parse natural language query, possibly asking clarification."""
    query = payload.get("query", "").strip()
    conversation = payload.get("conversation", None)
    if not query:
        raise HTTPException(400, "Query is required")

    engine = get_llm()
    result = engine.parse_query_interactive(query, conversation)
    return result


@app.post("/api/search")
async def search(payload: dict):
    query = payload.get("query", "").strip()
    if not query:
        raise HTTPException(400, "Query is required")

    engine = get_llm()

    # If intent provided (from two-step flow), use it directly
    if "intent" in payload:
        intent = SearchIntent(**payload["intent"])
    else:
        # Legacy one-shot fallback
        intent = engine.parse_query(query)

    # Deduplicate: remove search_terms that already appear in other fields
    all_specific = set(
        [v.lower() for v in intent.conditions]
        + [v.lower() for v in intent.interventions]
        + [v.lower() for v in intent.study_types]
    )
    if intent.search_terms and intent.search_terms.lower() in all_specific:
        intent.search_terms = ""

    # Step 2: Multi-strategy search for maximum recall
    # Build base search terms (without study types — those get their own strategy)
    search_parts = []
    if intent.search_terms:
        search_parts.append(intent.search_terms)
    if intent.conditions:
        search_parts.extend(intent.conditions)
    if intent.interventions:
        search_parts.extend(intent.interventions)
    base_query = " ".join(search_parts) if search_parts else query

    seen_ids: set[str] = set()
    all_results: list = []
    total = 0

    def _add_results(results: list, count: int):
        nonlocal total
        total = max(total, count)
        for r in results:
            if r.nct_id not in seen_ids:
                seen_ids.add(r.nct_id)
                all_results.append(r)

    # Strategy A: Broad search with base terms
    results_a, total_a = ctgov.search_studies(
        query_term=base_query,
        phases=intent.phases or None,
        statuses=intent.statuses or None,
        page_size=100,
        max_pages=3,
    )
    _add_results(results_a, total_a)

    # Strategy B: Study type focused search (if study types specified)
    if intent.study_types:
        expanded = expand_study_types(intent.study_types)
        # Build OR query with quoted synonym phrases (limit to top 12 to avoid too-long queries)
        or_terms = " OR ".join(f'"{t}"' for t in expanded[:12])
        study_type_query = f"{base_query} {or_terms}".strip() if base_query else or_terms
        results_b, total_b = ctgov.search_studies(
            query_term=study_type_query,
            phases=intent.phases or None,
            statuses=intent.statuses or None,
            page_size=100,
            max_pages=3,
        )
        _add_results(results_b, total_b)

    # Strategy C: Tight combined (base + primary study type together)
    if intent.study_types and base_query:
        tight_query = f"{base_query} {intent.study_types[0]}"
        results_c, total_c = ctgov.search_studies(
            query_term=tight_query,
            phases=intent.phases or None,
            statuses=intent.statuses or None,
            page_size=100,
            max_pages=2,
        )
        _add_results(results_c, total_c)

    # Step 3: Filter for studies with documents (only require ANY document)
    filtered = ctgov.filter_with_documents(
        all_results,
        require_protocol=False,
        require_sap=False,
        require_icf=False,
        min_enrollment=intent.min_enrollment,
    )

    # If too few results with docs, retry without status filter
    if len(filtered) < 5 and intent.statuses:
        more_results, _ = ctgov.search_studies(
            query_term=base_query,
            phases=intent.phases or None,
            statuses=None,
            page_size=100,
            max_pages=3,
        )
        more_filtered = ctgov.filter_with_documents(more_results)
        for s in more_filtered:
            if s.nct_id not in seen_ids:
                seen_ids.add(s.nct_id)
                filtered.append(s)

    # Step 3b: Local study type score boost (before LLM ranking)
    study_type_scores: dict[str, float] = {}
    if intent.study_types:
        for study in filtered:
            searchable = f"{study.brief_title} {study.official_title} {study.brief_summary}".strip()
            study_type_scores[study.nct_id] = score_study_type_match(searchable, intent.study_types)

    # Step 3c: CP Agent filter — remove non-CP-relevant studies
    cp_scored = filter_for_cp(filtered, min_cp_score=20)
    cp_info_map: dict[str, dict] = {}
    if cp_scored:
        # Use CP-filtered list if it has enough results; otherwise keep all
        if len(cp_scored) >= 3:
            filtered = [s for s, _ in cp_scored]
        # Always store CP info for display
        for s, info in cp_scored:
            cp_info_map[s.nct_id] = info

    # Step 4: Rank results with LLM
    ranked = engine.rank_results(query, filtered)

    # Apply study type boost after LLM ranking (+15 points max)
    if study_type_scores:
        for s in ranked:
            boost = study_type_scores.get(s.nct_id, 0.0) * 15
            if boost > 0 and s.relevance_score is not None:
                s.relevance_score = min(100, s.relevance_score + boost)

    # Apply CP score boost (+10 points max)
    for s in ranked:
        cp_info = cp_info_map.get(s.nct_id)
        if cp_info and s.relevance_score is not None:
            cp_boost = (cp_info["cp_score"] / 100) * 10
            s.relevance_score = min(100, s.relevance_score + cp_boost)

    ranked.sort(key=lambda x: x.relevance_score or 0, reverse=True)

    # Attach CP info to results for frontend display
    response = SearchResponse(
        query=query,
        intent=intent,
        results=ranked,
        total_found=total,
    ).model_dump()

    # Inject CP scores into response
    for r in response["results"]:
        nct = r["nct_id"]
        if nct in cp_info_map:
            r["cp_info"] = cp_info_map[nct]

    return response


@app.post("/api/cp-advise")
async def cp_advise(payload: dict):
    """CP Agent: given search results + research goal, recommend best protocols."""
    query = payload.get("query", "").strip()
    studies = payload.get("studies", [])
    if not query or not studies:
        raise HTTPException(400, "query and studies are required")

    import json
    engine = get_llm()

    # Build compact study summaries for the LLM
    summaries = []
    for s in studies[:10]:
        cp = s.get("cp_info", {})
        doc_types = []
        for d in s.get("documents", []):
            if d.get("has_protocol"): doc_types.append("Protocol")
            if d.get("has_sap"): doc_types.append("SAP")
            if d.get("has_icf"): doc_types.append("ICF")
        summaries.append({
            "nct_id": s["nct_id"],
            "title": s["brief_title"],
            "summary": (s.get("brief_summary") or "")[:300],
            "phases": s.get("phases", []),
            "enrollment": s.get("enrollment"),
            "conditions": s.get("conditions", [])[:3],
            "interventions": s.get("interventions", [])[:3],
            "documents": list(set(doc_types)),
            "cp_score": cp.get("cp_score", 0),
            "cp_flags": cp.get("flags", []),
            "cp_red_flags": cp.get("red_flags", []),
        })

    user_msg = (
        f"Researcher's goal: {query}\n\n"
        f"Candidate studies (sorted by CP relevance):\n{json.dumps(summaries, indent=2)}"
    )

    advice = engine.cp_advise(CP_ADVISOR_PROMPT, user_msg)
    return {"advice": advice}


@app.post("/api/download")
async def download(payload: dict):
    """Download a specific document and return it as a file."""
    url = payload.get("url", "")
    nct_id = payload.get("nct_id", "")
    if not url or not nct_id:
        raise HTTPException(400, "url and nct_id are required")

    download_dir = BASE_DIR / "downloads"
    try:
        local_path = ctgov.download_document(url, nct_id, str(download_dir))
    except Exception as e:
        raise HTTPException(500, f"Download failed: {e}")

    return FileResponse(
        local_path,
        media_type="application/pdf",
        filename=Path(local_path).name,
    )


@app.post("/api/preview")
async def preview(payload: dict):
    """Download protocol PDF, extract structured content, and generate LLM summary."""
    url = payload.get("url", "")
    nct_id = payload.get("nct_id", "")
    if not url or not nct_id:
        raise HTTPException(400, "url and nct_id are required")

    download_dir = BASE_DIR / "downloads"
    try:
        local_path = ctgov.download_document(url, nct_id, str(download_dir))
    except Exception as e:
        raise HTTPException(500, f"Download failed: {e}")

    # Extract structured content using docling (falls back to pdfplumber)
    try:
        content = extract_protocol(local_path)
    except Exception as e:
        raise HTTPException(500, f"PDF extraction failed: {e}")

    if not content.full_text.strip():
        return {
            "summary": "Could not extract text from this PDF (may be scanned/image-based).",
            "page_count": content.page_count,
            "sections": {},
            "tables": [],
            "has_redactions": False,
        }

    # Generate structured LLM summary using sections
    engine = get_llm()
    sections_text = format_sections_for_llm(content, max_chars=20000)

    if content.sections:
        # Use structured preview with section context
        summary = engine.preview_protocol_structured(sections_text)
    else:
        # Fallback to simple preview
        summary = engine.preview_protocol(content.full_text[:15000])

    return {
        "summary": summary,
        "page_count": content.page_count,
        "sections": {k: v[:500] + ("..." if len(v) > 500 else "") for k, v in content.sections.items()},
        "tables": content.tables[:5],  # First 5 tables
        "has_redactions": content.has_redactions,
        "section_names": list(content.sections.keys()),
    }


# ------------------------------------------------------------------
# Similarity Search
# ------------------------------------------------------------------
@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Serve the Upload Protocol page."""
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/api/similar")
async def find_similar(payload: dict):
    """Find trials similar to a given NCT ID."""
    nct_id = payload.get("nct_id", "").strip()
    top_k = payload.get("top_k", 15)
    if not nct_id:
        raise HTTPException(400, "nct_id is required")

    if not sim_engine.is_loaded:
        raise HTTPException(503, "Similarity index not built. Run: python3 build_trial_index.py")

    results = sim_engine.find_similar_by_nct(nct_id, top_k=top_k)

    # Enrich with CP scores
    from cp_agent import compute_cp_relevance
    from models import StudyResult, StudyDocument
    for r in results:
        # Build a minimal StudyResult for CP scoring
        docs = [StudyDocument(**d) for d in r.get("documents", [])] if r.get("documents") else []
        stub = StudyResult(
            nct_id=r["nct_id"],
            brief_title=r.get("brief_title", ""),
            official_title=r.get("official_title", ""),
            brief_summary=r.get("brief_summary", ""),
            phases=r.get("phases", []),
            enrollment=r.get("enrollment"),
            status=r.get("status", ""),
            conditions=r.get("conditions", []),
            interventions=r.get("interventions", []),
            sponsor=r.get("sponsor", ""),
            documents=docs,
        )
        cp = compute_cp_relevance(stub)
        r["cp_score"] = cp["cp_score"]
        r["cp_flags"] = cp["flags"]

    return {"nct_id": nct_id, "results": results, "index_size": sim_engine.total_indexed}


@app.post("/api/upload-protocol")
async def upload_protocol(file: UploadFile = File(...)):
    """Upload a protocol PDF, extract text, and find similar trials."""
    if not sim_engine.is_loaded:
        raise HTTPException(503, "Similarity index not built. Run: python3 build_trial_index.py")

    # Save uploaded file to temp location
    import tempfile
    suffix = Path(file.filename).suffix if file.filename else ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(BASE_DIR / "downloads")) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Extract structured content from PDF
        protocol = extract_protocol(tmp_path)

        if not protocol.full_text.strip():
            raise HTTPException(400, "Could not extract text from this PDF")

        # Build a rich text for embedding — combine key sections
        embed_parts = []
        if protocol.sections:
            for key in ["Synopsis", "Study Design", "Objectives", "Dosing", "Pharmacokinetics",
                        "Primary Endpoints", "Eligibility", "Drug-Drug Interaction", "Food Effect"]:
                if key in protocol.sections:
                    embed_parts.append(f"{key}: {protocol.sections[key][:2000]}")

        if not embed_parts:
            # Fallback to full text
            embed_parts.append(protocol.full_text[:15000])

        embed_text = "\n".join(embed_parts)

        # Extract condition/intervention hints from protocol for ontology matching
        upload_conditions = []
        upload_interventions = []
        for key in ["Conditions", "Indications", "Disease", "Indication"]:
            if key in protocol.sections:
                upload_conditions.append(protocol.sections[key][:500])
        for key in ["Drug", "Intervention", "Investigational Product", "Study Drug"]:
            if key in protocol.sections:
                upload_interventions.append(protocol.sections[key][:500])

        # Find similar trials (with ontology hints)
        results = sim_engine.find_similar_by_text(
            embed_text, top_k=15,
            conditions=upload_conditions or None,
            interventions=upload_interventions or None,
        )

        # Enrich with CP scores
        from cp_agent import compute_cp_relevance
        from models import StudyResult, StudyDocument
        for r in results:
            docs = [StudyDocument(**d) for d in r.get("documents", [])] if r.get("documents") else []
            stub = StudyResult(
                nct_id=r["nct_id"],
                brief_title=r.get("brief_title", ""),
                official_title=r.get("official_title", ""),
                brief_summary=r.get("brief_summary", ""),
                phases=r.get("phases", []),
                enrollment=r.get("enrollment"),
                status=r.get("status", ""),
                conditions=r.get("conditions", []),
                interventions=r.get("interventions", []),
                sponsor=r.get("sponsor", ""),
                documents=docs,
            )
            cp = compute_cp_relevance(stub)
            r["cp_score"] = cp["cp_score"]
            r["cp_flags"] = cp["flags"]

        # Extract some info from the protocol for display
        extracted_info = {
            "filename": file.filename,
            "page_count": protocol.page_count,
            "sections_found": list(protocol.sections.keys()),
            "tables_found": len(protocol.tables),
            "has_redactions": protocol.has_redactions,
        }

        return {"extracted_info": extracted_info, "results": results, "index_size": sim_engine.total_indexed}

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/api/similarity-status")
async def similarity_status():
    """Check if similarity index is loaded."""
    return {
        "loaded": sim_engine.is_loaded,
        "total_indexed": sim_engine.total_indexed,
    }


@app.post("/api/fda-reviews")
async def fda_reviews(payload: dict):
    """Look up FDA Clinical Pharmacology reviews for a study's interventions."""
    interventions = payload.get("interventions", [])
    nct_id = payload.get("nct_id", "")
    if not interventions:
        raise HTTPException(400, "interventions list is required")
    result = fda.get_fda_info_for_study(interventions)
    return {"nct_id": nct_id, **result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
