"""FDA Clinical Pharmacology Review client.

Queries openFDA APIs to find approved drug information, pharmacologic classes,
and links to FDA Clinical Pharmacology review PDFs for study interventions.

Data sources:
  - openFDA Drug Label API (api.fda.gov/drug/label.json)
  - openFDA Drugs@FDA API (api.fda.gov/drug/drugsfda.json)
  - FDA accessdata.fda.gov for direct CP review PDF URLs
"""
from __future__ import annotations

import re
import time
import logging
from typing import Any

import requests

from cp_agent import NON_CP_KEYWORDS

logger = logging.getLogger(__name__)

# Additional terms to skip when looking up drugs in FDA
SKIP_TERMS = [
    "placebo", "saline", "normal saline", "dextrose", "water",
    "standard of care", "standard care", "usual care",
    "no intervention", "observation", "watchful waiting",
    "sham", "mock",
] + NON_CP_KEYWORDS

# Dosage form words to strip from intervention names
DOSAGE_FORMS_RE = re.compile(
    r'\b(\d+\.?\d*\s*(mg|g|ml|mcg|ug|µg|iu|units?|mmol)(/\w+)?)\b'
    r'|(\b(tablets?|capsules?|injection|solution|suspension|powder|'
    r'concentrate|infusion|vial|syringe|patch|film|lozenge|inhaler)\b)',
    re.IGNORECASE,
)

# Parenthetical content like "(Brand Name)" or "(10 mg)"
PARENS_RE = re.compile(r'\([^)]*\)')


class FDAClient:
    """Wrapper for openFDA APIs with in-memory TTL caching."""

    FDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
    FDA_DRUGSFDA_URL = "https://api.fda.gov/drug/drugsfda.json"
    # Known suffixes for Clinical Pharmacology review PDFs
    CP_REVIEW_SUFFIXES = [
        "ClinPharmR.pdf",
        "ClinPharmR_Bioequiv.pdf",
        "ClinPharm.pdf",
    ]
    # Known suffixes for label PDFs
    LABEL_SUFFIXES = [
        "Lbl.pdf",
        "lbl.pdf",
    ]
    BASE_NDA_URL = "https://www.accessdata.fda.gov/drugsatfda_docs/nda"
    DRUGS_AT_FDA_URL = (
        "https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm"
        "?event=overview.process&varApplNo={nda_num}"
    )

    def __init__(self, timeout: int = 15, cache_ttl: int = 3600):
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CT-Protocol-Finder/1.0 (research tool)"
        })
        self._cache: dict[str, tuple[float, Any]] = {}
        self._last_request_time = 0.0

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------
    def _get_cached(self, key: str) -> Any | None:
        if key in self._cache:
            expiry, value = self._cache[key]
            if time.time() < expiry:
                return value
            del self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        self._cache[key] = (time.time() + self.cache_ttl, value)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.3:
            time.sleep(0.3 - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, params: dict) -> dict | None:
        """GET with rate limiting and retry on 429."""
        self._rate_limit()
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 429:
                logger.warning("FDA API rate limited, retrying in 2s...")
                time.sleep(2)
                resp = self.session.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"FDA API error: {e}")
            return None

    # ------------------------------------------------------------------
    # Drug name normalization
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_drug_name(name: str) -> str:
        """Clean up an intervention name for FDA lookup."""
        if not name:
            return ""
        text = name.strip()
        # Remove parenthetical content
        text = PARENS_RE.sub("", text)
        # Remove dosage forms and strengths
        text = DOSAGE_FORMS_RE.sub("", text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Lowercase
        text = text.lower()
        # If still multi-word and long, take first meaningful word
        # (e.g., "tofacitinib citrate" → keep as-is, FDA handles it)
        return text

    @staticmethod
    def _is_non_drug(name: str) -> bool:
        """Check if an intervention name is clearly NOT a drug."""
        lower = name.lower()
        for term in SKIP_TERMS:
            if term in lower:
                return True
        return False

    # ------------------------------------------------------------------
    # openFDA Drug Label API
    # ------------------------------------------------------------------
    def search_by_drug_name(self, drug_name: str) -> list[dict]:
        """Search openFDA label API for a drug by generic or brand name."""
        normalized = self.normalize_drug_name(drug_name)
        if not normalized or len(normalized) < 2:
            return []

        cache_key = f"label:{normalized}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Search substance_name first (most reliable — catches all formulations)
        data = self._get(self.FDA_LABEL_URL, {
            "search": f'openfda.substance_name:"{normalized}"',
            "limit": 10,
        })

        if not data or "results" not in data:
            # Fallback: search generic and brand name fields
            search_query = (
                f'openfda.generic_name:"{normalized}"'
                f'+openfda.brand_name:"{normalized}"'
            )
            data = self._get(self.FDA_LABEL_URL, {
                "search": search_query,
                "limit": 10,
            })

        results = []
        if data and "results" in data:
            seen_ndas = set()
            for item in data["results"]:
                openfda = item.get("openfda", {})
                nda_list = openfda.get("application_number", [])
                generic_names = openfda.get("generic_name", [])
                brand_names = openfda.get("brand_name", [])
                pharm_classes = openfda.get("pharm_class_epc", [])
                substances = openfda.get("substance_name", [])
                routes = openfda.get("route", [])

                # Extract CP narrative excerpt (shared across all NDAs)
                cp_text = ""
                cp_sections = item.get("clinical_pharmacology", [])
                if cp_sections:
                    cp_text = cp_sections[0][:500]

                # Emit one entry PER NDA/BLA in this label
                # (a single label can reference multiple application numbers)
                for nda in (nda_list if nda_list else [""]):
                    if nda and nda in seen_ndas:
                        continue
                    if nda:
                        seen_ndas.add(nda)
                    results.append({
                        "generic_name": generic_names[0] if generic_names else "",
                        "brand_name": brand_names[0] if brand_names else "",
                        "nda_number": nda,
                        "pharm_class": pharm_classes[0] if pharm_classes else "",
                        "substance_name": substances[0] if substances else "",
                        "route": routes[0] if routes else "",
                        "cp_narrative_excerpt": cp_text,
                    })

        self._set_cache(cache_key, results)
        return results

    # ------------------------------------------------------------------
    # openFDA Drugs@FDA API — review documents + TOC scraping
    # ------------------------------------------------------------------
    def get_review_docs(self, application_number: str) -> dict:
        """Get review documents for an NDA from Drugs@FDA API,
        then scrape the TOC page to find real CP review + label PDFs."""
        if not application_number:
            return {}

        cache_key = f"docs:{application_number}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Clean NDA number (remove "NDA", "ANDA", "BLA" prefix)
        nda_clean = (application_number
                     .replace("NDA", "").replace("ANDA", "")
                     .replace("BLA", "").strip())

        # Step 1: Query Drugs@FDA API for submission history
        data = self._get(self.FDA_DRUGSFDA_URL, {
            "search": f'application_number:"{application_number}"',
            "limit": 1,
        })

        api_docs = []
        approval_year = None
        orig_review_url = None
        if data and "results" in data and data["results"]:
            result = data["results"][0]
            for submission in result.get("submissions", []):
                sub_type = submission.get("submission_type", "")
                sub_status = submission.get("submission_status", "")
                sub_date = submission.get("submission_status_date", "")

                # Track approval year from ORIG submission
                if sub_type == "ORIG" and sub_status == "AP" and sub_date:
                    approval_year = sub_date[:4]

                for doc in submission.get("application_docs", []):
                    doc_type = doc.get("type", "")
                    doc_url = doc.get("url", "")
                    doc_date = doc.get("date", "")
                    if doc_type and doc_url:
                        api_docs.append({
                            "type": doc_type,
                            "url": doc_url,
                            "date": doc_date,
                            "submission_type": sub_type,
                        })
                        # Capture the ORIG Review link (leads to TOC)
                        if sub_type == "ORIG" and doc_type == "Review":
                            orig_review_url = doc_url

        # Step 2: Build TOC URL and scrape for real CP review + label links
        cp_review_url = None
        label_url = None
        review_toc_url = None
        drugs_at_fda_url = self.DRUGS_AT_FDA_URL.format(nda_num=nda_clean)

        if approval_year and nda_clean:
            # Try known TOC URL patterns
            for ext in ["TOC.html", "TOC.htm"]:
                toc_url = (
                    f"{self.BASE_NDA_URL}/{approval_year}/"
                    f"{nda_clean}Orig1s000{ext}"
                )
                links = self._scrape_toc_links(toc_url, nda_clean)
                if links:
                    review_toc_url = toc_url
                    cp_review_url = links.get("cp_review_url")
                    label_url = links.get("label_url")
                    break

            # Fallback: construct probable URLs and verify with HEAD
            if not review_toc_url:
                review_toc_url = (
                    f"{self.BASE_NDA_URL}/{approval_year}/"
                    f"{nda_clean}Orig1s000TOC.html"
                )
            if not cp_review_url:
                candidate = (
                    f"{self.BASE_NDA_URL}/{approval_year}/"
                    f"{nda_clean}Orig1s000ClinPharmR.pdf"
                )
                if self._url_exists(candidate):
                    cp_review_url = candidate
            if not label_url:
                candidate = (
                    f"{self.BASE_NDA_URL}/{approval_year}/"
                    f"{nda_clean}Orig1s000Lbl.pdf"
                )
                if self._url_exists(candidate):
                    label_url = candidate

        # Also try the ORIG review URL from the API as TOC fallback
        if not review_toc_url and orig_review_url:
            review_toc_url = orig_review_url

        result = {
            "docs": api_docs,
            "cp_review_url": cp_review_url,
            "label_url": label_url,
            "review_toc_url": review_toc_url,
            "drugs_at_fda_url": drugs_at_fda_url,
            "approval_year": approval_year,
        }
        self._set_cache(cache_key, result)
        return result

    def _url_exists(self, url: str) -> bool:
        """Check if a URL returns 200 via HEAD request."""
        try:
            resp = self.session.head(url, timeout=5, allow_redirects=True)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def _scrape_toc_links(self, toc_url: str, nda_clean: str) -> dict | None:
        """Scrape a TOC page to find CP review and label PDF links.

        The TOC pages use JavaScript to build links, but the link patterns
        are predictable: {NDA}Orig1s000ClinPharmR.pdf, {NDA}Orig1s000Lbl.pdf.
        We check both the HTML content and try HEAD requests for known patterns.
        """
        self._rate_limit()
        try:
            resp = self.session.get(toc_url, timeout=self.timeout)
            if resp.status_code != 200:
                return None
        except requests.RequestException:
            return None

        html = resp.text
        base_url = toc_url.rsplit("/", 1)[0]  # directory URL

        # Look for CP review link in HTML and verify it exists
        cp_review_url = None
        for suffix in self.CP_REVIEW_SUFFIXES:
            # Check if suffix appears in the page (JS-generated links)
            if suffix in html:
                candidate = f"{base_url}/{nda_clean}Orig1s000{suffix}"
                if self._url_exists(candidate):
                    cp_review_url = candidate
                    break

        # If not found in HTML, try known filenames with HEAD request
        if not cp_review_url:
            for suffix in self.CP_REVIEW_SUFFIXES:
                candidate = f"{base_url}/{nda_clean}Orig1s000{suffix}"
                if self._url_exists(candidate):
                    cp_review_url = candidate
                    break

        # Look for label link
        label_url = None
        for suffix in self.LABEL_SUFFIXES:
            if suffix in html:
                label_url = f"{base_url}/{nda_clean}Orig1s000{suffix}"
                break

        if not cp_review_url and not label_url:
            return None

        return {
            "cp_review_url": cp_review_url,
            "label_url": label_url,
        }

    # ------------------------------------------------------------------
    # Main entry point — get FDA info for a study's interventions
    # ------------------------------------------------------------------
    def get_fda_info_for_study(self, interventions: list[str]) -> dict:
        """Look up FDA data for a list of intervention names.

        Returns dict with:
            matches: list of drug match dicts
            has_fda_data: bool
            match_count: int
        """
        matches = []
        seen_ndas = set()

        for intervention in interventions:
            if self._is_non_drug(intervention):
                continue

            label_results = self.search_by_drug_name(intervention)
            if not label_results:
                continue

            for label in label_results:
                nda = label.get("nda_number", "")
                # Deduplicate by NDA
                if nda and nda in seen_ndas:
                    continue
                if nda:
                    seen_ndas.add(nda)

                # Get review docs for this NDA
                review_info = {}
                if nda:
                    review_info = self.get_review_docs(nda)

                # Filter review docs to show only "Review" and "Label" types
                review_docs = []
                if isinstance(review_info, dict):
                    for doc in review_info.get("docs", []):
                        if doc.get("type") in ("Review", "Label"):
                            review_docs.append(doc)

                # Extract URLs from review_info
                cp_review_url = None
                review_toc_url = None
                label_url = None
                drugs_at_fda_url = None
                if isinstance(review_info, dict):
                    cp_review_url = review_info.get("cp_review_url")
                    review_toc_url = review_info.get("review_toc_url")
                    label_url = review_info.get("label_url")
                    drugs_at_fda_url = review_info.get("drugs_at_fda_url")

                match = {
                    "query_name": intervention,
                    "generic_name": label.get("generic_name", ""),
                    "brand_name": label.get("brand_name", ""),
                    "nda_number": nda,
                    "pharm_class": label.get("pharm_class", ""),
                    "substance_name": label.get("substance_name", ""),
                    "route": label.get("route", ""),
                    "approval_status": "Approved" if nda else "Unknown",
                    "cp_review_url": cp_review_url,
                    "label_url": label_url,
                    "review_toc_url": review_toc_url,
                    "drugs_at_fda_url": drugs_at_fda_url,
                    "review_docs": review_docs[:5],
                    "cp_narrative_excerpt": label.get("cp_narrative_excerpt", ""),
                }
                matches.append(match)

        return {
            "matches": matches,
            "has_fda_data": len(matches) > 0,
            "match_count": len(matches),
        }
