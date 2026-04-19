"""FIH/Dose-Escalation Trial Design Taxonomy.

A structured vocabulary for classifying First-In-Human and dose-escalation
clinical trials across 7 design dimensions. Each dimension contains named
tags with regex patterns for automatic extraction from trial text.

This is a novel contribution — no published ontology captures FIH-specific
design elements at this level of granularity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Taxonomy: dimension -> tag -> list of regex patterns
# ---------------------------------------------------------------------------
FIH_TAXONOMY: dict[str, dict[str, list[str]]] = {
    "escalation_method": {
        "3+3": [r"\b3\s*\+\s*3\b", r"three\s*\+\s*three", r"three plus three"],
        "mTPI": [r"\bmTPI\b", r"modified toxicity probability interval"],
        "mTPI-2": [r"\bmTPI[\s-]*2\b"],
        "BOIN": [r"\bBOIN\b", r"bayesian optimal interval"],
        "CRM": [r"\bCRM\b", r"continual reassessment method"],
        "EWOC": [r"\bEWOC\b", r"escalation with overdose control"],
        "accelerated_titration": [r"accelerated titration", r"accelerated dose escalation"],
        "modified_fibonacci": [r"modified fibonacci", r"fibonacci dose"],
        "rolling_six": [r"rolling\s*(?:six|6)", r"rolling 6"],
        "i3+3": [r"\bi3\s*\+\s*3\b"],
        "rule_based": [r"rule[\s-]based\s+(?:dose\s+)?escalation"],
        "model_based": [r"model[\s-]based\s+(?:dose\s+)?escalation"],
    },
    "modality": {
        "small_molecule": [
            r"small molecule",
            r"\btablet\b",
            r"\bcapsule\b",
            r"oral.*(?:dose|mg|administration)",
            r"film[\s-]coated",
        ],
        "monoclonal_antibody": [
            r"monoclonal antibody",
            r"\bmAb\b",
            r"(?:fully\s+)?human(?:ized)?\s+antibody",
            r"anti[\s-]\w+\s+antibody",
        ],
        "ADC": [r"antibody[\s-]drug\s+conjugate", r"\bADC\b"],
        "bispecific": [r"bispecific", r"bi[\s-]specific"],
        "CAR_T": [r"CAR[\s-]?T", r"chimeric antigen receptor"],
        "siRNA": [r"\bsiRNA\b", r"small interfering RNA"],
        "antisense": [r"antisense\s*(?:oligonucleotide)?", r"\bASO\b"],
        "mRNA": [r"\bmRNA\b(?!\s*vaccine)"],
        "peptide": [r"\bpeptide\b(?!\s*vaccine)"],
        "fusion_protein": [r"fusion protein", r"Fc[\s-]fusion"],
        "biologic": [r"\bbiologic(?:al|s)?\b", r"\bbiosimilar\b"],
        "gene_therapy": [r"gene therapy", r"gene transfer", r"\bAAV\b"],
        "cell_therapy": [r"cell therapy", r"cellular therapy"],
        "radioligand": [r"radioligand", r"radiopharmaceutical", r"radiolabeled therapy"],
        "degrader": [r"\bPROTAC\b", r"molecular glue", r"protein degrader"],
    },
    "pk_design": {
        "SAD": [r"\bSAD\b", r"single ascending dose", r"single[\s-]dose escalation"],
        "MAD": [r"\bMAD\b", r"multiple ascending dose", r"multiple[\s-]dose escalation",
                r"repeat[\s-]dose escalation"],
        "food_effect": [r"food effect", r"fed[\s/]fasted", r"fed\s+(?:and|&)\s+fasted",
                        r"high[\s-]fat\s+meal", r"effect of food"],
        "DDI": [r"drug[\s-]drug\s+interaction", r"\bDDI\b(?:\s+study)?"],
        "bioavailability": [r"bioavailability", r"\bBA\s+study\b", r"relative\s+BA",
                            r"absolute\s+BA"],
        "bioequivalence": [r"bioequivalence", r"\bBE\s+study\b"],
        "mass_balance": [r"mass balance", r"\bADME\b", r"radiolabel(?:ed|led)",
                         r"carbon[\s-]14", r"\b14C\b", r"\[14C\]"],
        "QTc": [r"\bQTc?\b(?:\s+(?:study|prolongation))?", r"thorough\s+QT", r"\bTQT\b",
                r"cardiac\s+safety", r"QTcF"],
        "renal_impairment": [r"renal\s+impairment", r"kidney\s+impairment",
                             r"renal\s+(?:in)?sufficiency"],
        "hepatic_impairment": [r"hepatic\s+impairment", r"liver\s+impairment",
                               r"hepatic\s+(?:in)?sufficiency", r"Child[\s-]Pugh"],
        "dose_proportionality": [r"dose\s+proportionality", r"dose[\s-]proportional"],
    },
    "dlt_window": {
        "21_day": [r"21[\s-]day", r"3[\s-]week.*(?:DLT|observation|assessment|window)"],
        "28_day": [r"28[\s-]day", r"4[\s-]week.*(?:DLT|observation|assessment|window)",
                   r"cycle\s+1.*28\s*day"],
        "42_day": [r"42[\s-]day", r"6[\s-]week.*(?:DLT|observation|assessment|window)"],
        "cycle_1": [r"(?:during|in|within)\s+cycle\s*1", r"first\s+(?:treatment\s+)?cycle"],
    },
    "population": {
        "healthy_volunteers": [
            r"healthy\s+(?:volunteer|subject|participant|adult|male|female|individual)s?",
            r"\bHV\b(?:\s+subject)?",
            r"healthy\s+(?:men|women)",
        ],
        "patients": [
            r"(?:advanced|metastatic|relapsed|refractory)\s+(?:solid\s+)?(?:cancer|tumor|tumour|malignancy|malignancies)",
            r"(?:cancer|tumor)\s+patients?",
        ],
        "japanese_bridging": [
            r"japanese\s+(?:subject|patient|bridging|pharmacokinetic)s?",
            r"ethnic\s+sensitivity",
            r"asian\s+(?:subject|patient)s?",
        ],
        "pediatric": [r"pediatric", r"paediatric", r"\bchildren\b", r"\badolescent\b"],
        "elderly": [r"\belderly\b", r"\bgeriatric\b", r"older\s+adult"],
        "renal_impaired": [r"renal(?:ly)?\s+impair(?:ed|ment)", r"kidney\s+impair"],
        "hepatic_impaired": [r"hepatic(?:ally)?\s+impair(?:ed|ment)", r"liver\s+impair",
                             r"Child[\s-]Pugh"],
        "obese": [r"\bobese\b", r"\bobesity\b", r"BMI\s*(?:>|>=|above)\s*30"],
    },
    "pd_biomarkers": {
        "receptor_occupancy": [r"receptor\s+occupancy", r"\bRO\b\s*(?:assay|measurement|%)"],
        "target_engagement": [r"target\s+engagement"],
        "cytokine_panel": [r"cytokine", r"interleukin", r"\bIL[\s-]\d"],
        "pk_pd_modeling": [r"PK/?PD", r"exposure[\s-]response", r"pharmacokinetic[\s/]pharmacodynamic"],
        "immunogenicity": [r"immunogenicity", r"anti[\s-]drug\s+antibod", r"\bADA\b"],
        "biomarker_general": [r"(?:pharmacodynamic|PD)\s+(?:bio)?marker", r"surrogate\s+endpoint"],
    },
    "route": {
        "IV": [r"\bIV\b", r"intravenous", r"\binfusion\b"],
        "SC": [r"\bSC\b", r"subcutaneous"],
        "oral": [r"\boral(?:ly)?\b", r"\btablet\b", r"\bcapsule\b"],
        "IM": [r"\bIM\b", r"intramuscular"],
        "intrathecal": [r"intrathecal"],
        "inhaled": [r"inhal(?:ed|ation)", r"\bICS\b", r"\bDPI\b"],
        "topical": [r"\btopical\b", r"transdermal"],
        "ophthalmic": [r"ophthalmic", r"intravitreal", r"\bIVT\b"],
        "intranasal": [r"intranasal"],
    },
}

# Dimension weights for similarity computation
DIMENSION_WEIGHTS: dict[str, float] = {
    "pk_design": 0.25,
    "modality": 0.20,
    "escalation_method": 0.15,
    "population": 0.15,
    "route": 0.10,
    "dlt_window": 0.08,
    "pd_biomarkers": 0.07,
}


# ---------------------------------------------------------------------------
# FIH Profile dataclass
# ---------------------------------------------------------------------------
@dataclass
class FIHProfile:
    """Extracted FIH design profile for a clinical trial."""
    escalation_method: list[str] = field(default_factory=list)
    modality: list[str] = field(default_factory=list)
    pk_design: list[str] = field(default_factory=list)
    dlt_window: list[str] = field(default_factory=list)
    population: list[str] = field(default_factory=list)
    pd_biomarkers: list[str] = field(default_factory=list)
    route: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "escalation_method": self.escalation_method,
            "modality": self.modality,
            "pk_design": self.pk_design,
            "dlt_window": self.dlt_window,
            "population": self.population,
            "pd_biomarkers": self.pd_biomarkers,
            "route": self.route,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FIHProfile:
        return cls(
            escalation_method=d.get("escalation_method", []),
            modality=d.get("modality", []),
            pk_design=d.get("pk_design", []),
            dlt_window=d.get("dlt_window", []),
            population=d.get("population", []),
            pd_biomarkers=d.get("pd_biomarkers", []),
            route=d.get("route", []),
        )

    def summary(self) -> str:
        """Human-readable summary of the profile."""
        parts = []
        if self.pk_design:
            parts.append("+".join(self.pk_design))
        if self.escalation_method:
            parts.append(", ".join(self.escalation_method))
        if self.modality:
            parts.append(", ".join(self.modality))
        if self.route:
            parts.append(", ".join(self.route))
        if self.population:
            parts.append(", ".join(self.population))
        return " | ".join(parts) if parts else "No FIH features detected"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_fih_profile(text: str) -> FIHProfile:
    """Extract structured FIH profile from trial text using regex patterns.

    Args:
        text: Combined trial text (title + official_title + brief_summary).

    Returns:
        FIHProfile with matched tags for each dimension.
    """
    if not text:
        return FIHProfile()

    text_lower = text.lower()
    profile_data: dict[str, list[str]] = {}

    for dimension, tags in FIH_TAXONOMY.items():
        matched_tags: list[str] = []
        for tag_name, patterns in tags.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        matched_tags.append(tag_name)
                        break  # One match per tag is enough
                except re.error:
                    continue
        profile_data[dimension] = matched_tags

    return FIHProfile.from_dict(profile_data)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity: |intersection| / |union|.

    Returns:
        0.5 if both empty (neutral — no information),
        0.0 if one empty and other non-empty,
        Jaccard index otherwise.
    """
    if not set_a and not set_b:
        return 0.5  # Both empty — neutral
    if not set_a or not set_b:
        return 0.0  # One has info, other doesn't — no match
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def fih_profile_similarity(profile_a: FIHProfile, profile_b: FIHProfile) -> float:
    """Compute weighted similarity between two FIH profiles.

    Each dimension is scored independently using Jaccard similarity,
    then dimensions are combined using DIMENSION_WEIGHTS.

    Returns:
        Float 0.0 to 1.0 where 1.0 means identical profiles.
    """
    total_score = 0.0
    total_weight = 0.0

    for dimension, weight in DIMENSION_WEIGHTS.items():
        tags_a = set(getattr(profile_a, dimension, []))
        tags_b = set(getattr(profile_b, dimension, []))
        dim_score = _jaccard_similarity(tags_a, tags_b)
        total_score += weight * dim_score
        total_weight += weight

    if total_weight == 0:
        return 0.5

    return total_score / total_weight


def fih_dimension_breakdown(profile_a: FIHProfile, profile_b: FIHProfile) -> dict[str, float]:
    """Per-dimension similarity breakdown."""
    breakdown = {}
    for dimension in DIMENSION_WEIGHTS:
        tags_a = set(getattr(profile_a, dimension, []))
        tags_b = set(getattr(profile_b, dimension, []))
        breakdown[dimension] = _jaccard_similarity(tags_a, tags_b)
    return breakdown
