"""ClinicalTrials.gov API v2 client — search studies and download documents."""
from __future__ import annotations

import os
from pathlib import Path

import requests

from models import StudyDocument, StudyResult

BASE_URL = "https://clinicaltrials.gov/api/v2"
CDN_URL = "https://cdn.clinicaltrials.gov/large-docs"

# Fields we request from the API
FIELDS = "|".join([
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.identificationModule.officialTitle",
    "protocolSection.designModule.phases",
    "protocolSection.designModule.enrollmentInfo",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.armsInterventionsModule.interventions",
    "protocolSection.sponsorCollaboratorsModule.leadSponsor",
    "protocolSection.descriptionModule.briefSummary",
    "protocolSection.outcomesModule.primaryOutcomes",
    "documentSection.largeDocumentModule",
])


class CTGovClient:
    """Wrapper around the ClinicalTrials.gov v2 API."""

    def __init__(self, timeout: int = 30):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "CTProtocolFinder/1.0"
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search_studies(
        self,
        query_term: str = "",
        conditions: list[str] | None = None,
        interventions: list[str] | None = None,
        phases: list[str] | None = None,
        statuses: list[str] | None = None,
        page_size: int = 100,
        max_pages: int = 3,
    ) -> tuple[list[StudyResult], int]:
        """Search ClinicalTrials.gov and return parsed results."""
        params: dict = {"fields": FIELDS, "pageSize": min(page_size, 100)}

        # Build query.term — combine free text with phase filters via AREA[] syntax
        term_parts = []
        if query_term:
            term_parts.append(query_term)
        if phases:
            phase_expr = " OR ".join(f"AREA[Phase]{p}" for p in phases)
            term_parts.append(f"({phase_expr})")
        if term_parts:
            params["query.term"] = " ".join(term_parts)
        if conditions:
            params["query.cond"] = ",".join(conditions)
        if interventions:
            params["query.intr"] = ",".join(interventions)
        if statuses:
            params["filter.overallStatus"] = ",".join(statuses)

        all_results: list[StudyResult] = []
        total_count = 0

        for _ in range(max_pages):
            resp = self.session.get(
                f"{BASE_URL}/studies", params=params, timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()

            total_count = data.get("totalCount", 0)
            studies = data.get("studies", [])

            for raw in studies:
                study = self._parse_study(raw)
                if study:
                    all_results.append(study)

            # Pagination
            next_token = data.get("nextPageToken")
            if not next_token or not studies:
                break
            params["pageToken"] = next_token

        return all_results, total_count

    # ------------------------------------------------------------------
    # Single study lookup
    # ------------------------------------------------------------------
    def get_study(self, nct_id: str) -> StudyResult | None:
        resp = self.session.get(
            f"{BASE_URL}/studies/{nct_id}",
            params={"fields": FIELDS},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return self._parse_study(resp.json())

    # ------------------------------------------------------------------
    # Parse raw API response → StudyResult
    # ------------------------------------------------------------------
    def _parse_study(self, raw: dict) -> StudyResult | None:
        proto = raw.get("protocolSection", {})
        if not proto:
            return None

        ident = proto.get("identificationModule", {})
        design = proto.get("designModule", {})
        status_mod = proto.get("statusModule", {})
        cond_mod = proto.get("conditionsModule", {})
        arms_mod = proto.get("armsInterventionsModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})

        nct_id = ident.get("nctId", "")
        if not nct_id:
            return None

        # Extract brief summary + primary outcomes for study type matching
        desc_mod = proto.get("descriptionModule", {})
        brief_summary = desc_mod.get("briefSummary", "")
        outcomes_mod = proto.get("outcomesModule", {})
        primary_outcomes = outcomes_mod.get("primaryOutcomes", [])
        outcome_texts = [o.get("measure", "") for o in primary_outcomes if o.get("measure")]
        if outcome_texts:
            brief_summary += " | Outcomes: " + "; ".join(outcome_texts)

        # Extract interventions names
        intervention_names = []
        for intr in arms_mod.get("interventions", []):
            name = intr.get("name", "")
            if name:
                intervention_names.append(name)

        # Extract enrollment
        enrollment_info = design.get("enrollmentInfo", {})
        enrollment = enrollment_info.get("count") if isinstance(enrollment_info, dict) else None

        # Parse documents
        doc_section = raw.get("documentSection", {})
        large_doc_mod = doc_section.get("largeDocumentModule", {})
        large_docs = large_doc_mod.get("largeDocs", [])

        documents = []
        for doc in large_docs:
            filename = doc.get("filename", "")
            if not filename:
                continue
            # Construct CDN download URL
            suffix = nct_id[-2:]
            download_url = f"{CDN_URL}/{suffix}/{nct_id}/{filename}"
            documents.append(StudyDocument(
                has_protocol=doc.get("hasProtocol", False),
                has_sap=doc.get("hasSap", False),
                has_icf=doc.get("hasIcf", False),
                label=doc.get("label", ""),
                filename=filename,
                date=doc.get("date", ""),
                size=doc.get("size", 0),
                download_url=download_url,
            ))

        return StudyResult(
            nct_id=nct_id,
            brief_title=ident.get("briefTitle", ""),
            official_title=ident.get("officialTitle", ""),
            brief_summary=brief_summary,
            phases=design.get("phases", []),
            enrollment=enrollment,
            status=status_mod.get("overallStatus", ""),
            conditions=cond_mod.get("conditions", []),
            interventions=intervention_names,
            sponsor=sponsor_mod.get("leadSponsor", {}).get("name", ""),
            documents=documents,
        )

    # ------------------------------------------------------------------
    # Filter for studies that have specific document types
    # ------------------------------------------------------------------
    @staticmethod
    def filter_with_documents(
        studies: list[StudyResult],
        require_protocol: bool = False,
        require_sap: bool = False,
        require_icf: bool = False,
        min_enrollment: int | None = None,
    ) -> list[StudyResult]:
        filtered = []
        for s in studies:
            if min_enrollment and (s.enrollment is None or s.enrollment < min_enrollment):
                continue
            if require_protocol and not any(d.has_protocol for d in s.documents):
                continue
            if require_sap and not any(d.has_sap for d in s.documents):
                continue
            if require_icf and not any(d.has_icf for d in s.documents):
                continue
            # Must have at least one document
            if not s.documents:
                continue
            filtered.append(s)
        return filtered

    # ------------------------------------------------------------------
    # Download a document PDF
    # ------------------------------------------------------------------
    def download_document(
        self, download_url: str, nct_id: str, output_dir: str = "downloads"
    ) -> str:
        """Download a PDF and return the local file path."""
        save_dir = Path(output_dir) / nct_id
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = download_url.split("/")[-1]
        save_path = save_dir / filename

        if save_path.exists():
            return str(save_path)

        resp = self.session.get(download_url, stream=True, timeout=60)
        resp.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return str(save_path)
