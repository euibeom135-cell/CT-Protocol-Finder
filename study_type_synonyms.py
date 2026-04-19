"""Study type synonym expansion and local text-matching for clinical trial search."""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Canonical study type → synonym variants
# These appear in trial titles, summaries, objectives, and endpoints.
# ---------------------------------------------------------------------------
STUDY_TYPE_SYNONYMS: dict[str, list[str]] = {
    "first in human": [
        "first-in-human", "FIH", "first in man", "first-in-man",
        "first time in human", "first human dose",
    ],
    "single ascending dose": [
        "SAD", "single dose escalation", "single-ascending-dose",
        "single dose", "ascending single dose",
    ],
    "multiple ascending dose": [
        "MAD", "multiple dose escalation", "multiple-ascending-dose",
        "multiple dose", "ascending multiple dose", "repeat dose",
    ],
    "dose escalation": [
        "dose-escalation", "dose finding", "dose-finding",
        "dose ranging", "dose-ranging", "dose exploration",
    ],
    "food effect": [
        "food-effect", "fed fasted", "fed-fasted", "fed and fasted",
        "effect of food", "food interaction", "high-fat meal",
    ],
    "drug-drug interaction": [
        "DDI", "drug interaction", "drug-drug", "drug interaction study",
        "co-administration", "coadministration",
    ],
    "pharmacokinetics": [
        "PK", "pharmacokinetic", "PK study", "pharmacokinetic study",
        "PK profile", "pharmacokinetic profile", "absorption",
    ],
    "pharmacodynamics": [
        "PD", "pharmacodynamic", "PK/PD", "PK-PD",
        "pharmacokinetic/pharmacodynamic",
    ],
    "bioequivalence": [
        "BE study", "bioequivalence study", "bio-equivalence",
        "BE", "bioequivalent",
    ],
    "relative bioavailability": [
        "relative BA", "relative bioavailability study",
        "formulation comparison", "formulation bridging",
    ],
    "absolute bioavailability": [
        "absolute BA", "IV microtracer", "intravenous microtracer",
    ],
    "mass balance": [
        "ADME", "absorption distribution metabolism excretion",
        "mass balance study", "radiolabeled", "radiolabelled",
        "carbon-14", "14C", "[14C]",
    ],
    "QTc": [
        "QT prolongation", "thorough QT", "TQT", "QTc study",
        "cardiac safety", "QTcF", "electrocardiogram",
    ],
    "renal impairment": [
        "renal impairment study", "kidney impairment",
        "renal function", "renal insufficiency",
    ],
    "hepatic impairment": [
        "hepatic impairment study", "liver impairment",
        "hepatic function", "hepatic insufficiency", "Child-Pugh",
    ],
    "thorough QT": [
        "TQT", "thorough QT study", "QTc prolongation",
    ],
    "safety and tolerability": [
        "safety tolerability", "safety/tolerability",
        "tolerability study", "safety study",
    ],
    "immunogenicity": [
        "anti-drug antibody", "ADA", "immunogenicity assessment",
    ],
    "pediatric": [
        "paediatric", "pediatric study", "children", "adolescent",
    ],
    "Japanese bridging": [
        "Japanese subjects", "ethnic sensitivity", "bridging study Japan",
    ],
    "organ impairment": [
        "special population", "special populations",
        "organ impairment study",
    ],
}


def expand_study_types(study_types: list[str]) -> list[str]:
    """Given canonical study type names, return all synonym variants (including the canonical name).

    Returns a flat deduplicated list.
    """
    expanded: list[str] = []
    seen: set[str] = set()

    for st in study_types:
        st_lower = st.lower().strip()
        # Add the canonical name itself
        if st_lower not in seen:
            expanded.append(st_lower)
            seen.add(st_lower)
        # Add synonyms
        for canonical, synonyms in STUDY_TYPE_SYNONYMS.items():
            if canonical == st_lower or st_lower in [s.lower() for s in synonyms]:
                # Found a match — add all variants
                if canonical not in seen:
                    expanded.append(canonical)
                    seen.add(canonical)
                for syn in synonyms:
                    syn_lower = syn.lower()
                    if syn_lower not in seen:
                        expanded.append(syn_lower)
                        seen.add(syn_lower)

    return expanded


def score_study_type_match(text: str, study_types: list[str]) -> float:
    """Score how well a text matches the requested study types.

    Returns 0.0 to 1.0 based on what fraction of requested study types
    have at least one synonym match in the text.
    """
    if not study_types or not text:
        return 0.0

    text_lower = text.lower()
    matched = 0

    for st in study_types:
        st_lower = st.lower().strip()
        # Check canonical name
        if st_lower in text_lower:
            matched += 1
            continue
        # Check synonyms
        found = False
        for canonical, synonyms in STUDY_TYPE_SYNONYMS.items():
            if canonical == st_lower or st_lower in [s.lower() for s in synonyms]:
                # Check all variants against text
                all_variants = [canonical] + [s.lower() for s in synonyms]
                for variant in all_variants:
                    # Use word boundary matching for short acronyms (<=3 chars)
                    if len(variant) <= 3:
                        if re.search(r'\b' + re.escape(variant) + r'\b', text_lower):
                            found = True
                            break
                    else:
                        if variant in text_lower:
                            found = True
                            break
                if found:
                    break
        if found:
            matched += 1

    return matched / len(study_types)
