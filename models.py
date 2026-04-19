from __future__ import annotations
from pydantic import BaseModel, Field


class SearchIntent(BaseModel):
    """LLM-parsed interpretation of a user's natural language query."""
    search_terms: str = ""
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    study_types: list[str] = Field(default_factory=list)  # first in human, dose escalation, food effect, etc.
    phases: list[str] = Field(default_factory=list)  # PHASE1, PHASE2, PHASE3, PHASE4, EARLY_PHASE1
    statuses: list[str] = Field(default_factory=list)
    require_protocol: bool = True
    require_sap: bool = False
    require_icf: bool = False
    min_enrollment: int | None = None
    rationale: str = ""


class ParseResponse(BaseModel):
    """Response from the parse step -- either a resolved intent or clarification questions."""
    status: str = "ok"  # "ok" | "clarify"
    intent: SearchIntent | None = None
    questions: list[str] = Field(default_factory=list)
    rationale: str = ""


class StudyDocument(BaseModel):
    """A single downloadable document from a clinical trial."""
    has_protocol: bool = False
    has_sap: bool = False
    has_icf: bool = False
    label: str = ""
    filename: str = ""
    date: str = ""
    size: int = 0
    download_url: str = ""


class StudyResult(BaseModel):
    """A clinical trial study with its metadata and available documents."""
    nct_id: str
    brief_title: str = ""
    official_title: str = ""
    phases: list[str] = Field(default_factory=list)
    enrollment: int | None = None
    status: str = ""
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    sponsor: str = ""
    documents: list[StudyDocument] = Field(default_factory=list)
    relevance_score: float | None = None
    relevance_explanation: str = ""
    brief_summary: str = ""  # From API: study description for local matching
    preview_summary: str = ""  # On-demand LLM summary of protocol content


class ProtocolContent(BaseModel):
    """Structured content extracted from a protocol PDF using docling."""
    full_text: str = ""
    sections: dict[str, str] = Field(default_factory=dict)  # "Study Design" → text
    tables: list[str] = Field(default_factory=list)  # markdown tables
    page_count: int = 0
    has_redactions: bool = False
    source_file: str = ""


class SimilarTrial(BaseModel):
    """A trial found via similarity search."""
    nct_id: str
    brief_title: str = ""
    similarity_score: float = 0.0
    brief_summary: str = ""
    phases: list[str] = Field(default_factory=list)
    enrollment: int | None = None
    status: str = ""
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    sponsor: str = ""
    documents: list[StudyDocument] = Field(default_factory=list)
    cp_score: float = 0.0


class FDADrugMatch(BaseModel):
    """A single FDA drug match for an intervention."""
    query_name: str = ""
    generic_name: str = ""
    brand_name: str = ""
    nda_number: str = ""
    pharm_class: str = ""
    substance_name: str = ""
    route: str = ""
    approval_status: str = ""
    cp_review_url: str | None = None
    review_toc_url: str | None = None
    review_docs: list[dict] = Field(default_factory=list)
    cp_narrative_excerpt: str = ""


class SearchResponse(BaseModel):
    """Full response returned to the frontend."""
    query: str
    intent: SearchIntent
    results: list[StudyResult] = Field(default_factory=list)
    total_found: int = 0
