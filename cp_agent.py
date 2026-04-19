"""Clinical Pharmacology Agent — domain expert filter for protocol/SAP selection.

This agent understands what a clinical pharmacologist needs:
- PK-rich study designs (FIH, SAD/MAD, dose escalation, food effect, DDI, etc.)
- Systemically absorbed drugs (oral, IV, SC) — NOT topicals, vaccines, devices
- Protocols with real PK data: concentration-time curves, compartmental modeling, etc.
- SAPs with proper statistical methodology for PK endpoints

It scores and filters search results through a CP lens, removing noise
(vaccines, ointments, devices, behavioral interventions) and surfacing
the most data-rich protocols for clinical pharmacology research.
"""
from __future__ import annotations

import re

from models import StudyResult

# ---------------------------------------------------------------------------
# Drug modality scoring — how relevant is this modality for CP/PK research?
# Higher = more relevant for PK analysis
# ---------------------------------------------------------------------------
MODALITY_SCORES: dict[str, float] = {
    # Highly relevant — systemic PK, rich data
    "small molecule oral": 1.0,
    "small molecule": 1.0,
    "tablet": 1.0,
    "capsule": 1.0,
    "oral solution": 0.95,
    "oral suspension": 0.95,
    "iv infusion": 0.95,
    "intravenous": 0.95,
    "subcutaneous": 0.90,
    "intramuscular": 0.85,
    "monoclonal antibody": 0.80,
    "biologic": 0.75,
    "peptide": 0.80,
    "fusion protein": 0.75,
    "antibody-drug conjugate": 0.80,
    "adc": 0.80,
    # Moderate — some PK relevance
    "transdermal": 0.50,
    "inhaled": 0.55,
    "nasal": 0.50,
    "sublingual": 0.60,
    "ophthalmic": 0.30,
    # Low relevance — minimal systemic PK
    "topical": 0.15,
    "ointment": 0.10,
    "cream": 0.15,
    "gel": 0.15,
    "lotion": 0.10,
    # Not CP-relevant
    "vaccine": 0.05,
    "device": 0.0,
    "behavioral": 0.0,
    "dietary supplement": 0.05,
    "radiation": 0.0,
    "surgery": 0.0,
    "physical therapy": 0.0,
    "psychotherapy": 0.0,
    "placebo": 0.0,  # placebo alone isn't interesting
}

# Intervention keywords that signal NON-CP-relevant studies
NON_CP_KEYWORDS = [
    "vaccine", "vaccination", "immunization",
    "ointment", "cream", "lotion", "gel", "topical",
    "device", "implant", "stent",
    "behavioral", "cognitive behavioral", "counseling", "psychotherapy",
    "dietary", "exercise", "physical therapy", "physiotherapy",
    "radiation", "radiotherapy", "brachytherapy",
    "surgery", "surgical", "transplant",
    "acupuncture", "massage",
]

# Study design keywords that signal HIGH CP relevance
CP_RICH_DESIGN_KEYWORDS = {
    # FIH / dose escalation — gold standard for CP
    "first in human": 10,
    "first-in-human": 10,
    "fih": 10,
    "single ascending dose": 10,
    "sad": 8,
    "multiple ascending dose": 10,
    "mad": 8,
    "dose escalation": 9,
    "dose-escalation": 9,
    "dose finding": 8,
    "dose ranging": 8,
    # PK-focused designs
    "pharmacokinetic": 9,
    "pharmacokinetics": 9,
    "bioavailability": 8,
    "bioequivalence": 8,
    "food effect": 9,
    "fed fasted": 9,
    "drug-drug interaction": 8,
    "drug interaction": 7,
    "mass balance": 9,
    "adme": 9,
    "radiolabeled": 8,
    "absorption": 7,
    "metabolism": 7,
    # PK/PD
    "pharmacodynamic": 7,
    "pk/pd": 8,
    "exposure-response": 8,
    "concentration-time": 9,
    # Special populations (PK focus)
    "renal impairment": 7,
    "hepatic impairment": 7,
    "qtc": 7,
    "thorough qt": 8,
    "cardiac safety": 6,
    # Safety/tolerability in early phase
    "safety and tolerability": 5,
    "maximum tolerated dose": 8,
    "mtd": 7,
}


def compute_cp_relevance(study: StudyResult) -> dict:
    """Score a study for Clinical Pharmacology relevance.

    Returns dict with:
      - cp_score: 0-100 overall CP relevance
      - modality_score: 0-1 drug modality relevance
      - design_score: 0-10 study design relevance
      - flags: list of str reasons (green flags and red flags)
      - is_cp_relevant: bool — should this be shown to a CP researcher?
    """
    # Combine all searchable text
    text = " ".join([
        study.brief_title or "",
        study.official_title or "",
        study.brief_summary or "",
        " ".join(study.interventions),
        " ".join(study.conditions),
    ]).lower()

    flags: list[str] = []
    red_flags: list[str] = []

    # --- 1. Modality scoring ---
    modality_score = 0.5  # default: unknown modality
    for modality, score in MODALITY_SCORES.items():
        if modality in text:
            if score > modality_score:
                modality_score = score
            if score < 0.2:
                red_flags.append(f"Low CP relevance: {modality}")

    # Check interventions specifically for non-CP keywords
    interventions_text = " ".join(study.interventions).lower()
    for kw in NON_CP_KEYWORDS:
        if kw in interventions_text:
            red_flags.append(f"Non-CP intervention: {kw}")
            modality_score = min(modality_score, 0.2)

    # --- 2. Study design scoring ---
    design_score = 0
    design_matches = []
    for keyword, score in CP_RICH_DESIGN_KEYWORDS.items():
        # Use word boundary for short keywords
        if len(keyword) <= 4:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                if score > design_score:
                    design_score = score
                design_matches.append(keyword)
        else:
            if keyword in text:
                if score > design_score:
                    design_score = score
                design_matches.append(keyword)

    if design_matches:
        flags.append(f"CP design: {', '.join(design_matches[:3])}")

    # --- 3. Phase bonus ---
    phase_bonus = 0
    phases_text = " ".join(study.phases).lower()
    if "phase1" in phases_text or "early_phase1" in phases_text:
        phase_bonus = 2  # Phase 1 = most CP-relevant
        flags.append("Phase 1 (ideal for CP)")
    elif "phase2" in phases_text:
        phase_bonus = 1
    # Phase 3/4 get no bonus — less CP-rich typically

    # --- 4. Document bonus ---
    doc_bonus = 0
    has_protocol = any(d.has_protocol for d in study.documents)
    has_sap = any(d.has_sap for d in study.documents)
    if has_protocol and has_sap:
        doc_bonus = 2
        flags.append("Has both Protocol + SAP")
    elif has_protocol:
        doc_bonus = 1
        flags.append("Has Protocol")

    # --- 5. Enrollment consideration ---
    enrollment_factor = 1.0
    if study.enrollment:
        if 10 <= study.enrollment <= 200:
            enrollment_factor = 1.1  # Sweet spot for early-phase PK
            flags.append(f"Good CP enrollment ({study.enrollment})")
        elif study.enrollment > 1000:
            enrollment_factor = 0.8  # Likely Phase 3 efficacy trial

    # --- Compute final CP score ---
    # Weighted combination: design (50%) + modality (30%) + phase (10%) + docs (10%)
    raw_score = (
        (design_score / 10) * 50 +
        modality_score * 30 +
        phase_bonus * 5 +
        doc_bonus * 5
    ) * enrollment_factor

    cp_score = min(100, max(0, round(raw_score)))

    # Determine if CP-relevant (threshold: 25)
    is_relevant = cp_score >= 25 and len(red_flags) < 2

    return {
        "cp_score": cp_score,
        "modality_score": round(modality_score, 2),
        "design_score": design_score,
        "flags": flags,
        "red_flags": red_flags,
        "is_cp_relevant": is_relevant,
    }


def filter_for_cp(studies: list[StudyResult], min_cp_score: int = 25) -> list[tuple[StudyResult, dict]]:
    """Filter and score studies for CP relevance.

    Returns list of (study, cp_info) tuples, sorted by cp_score descending.
    Only includes studies above the min_cp_score threshold.
    """
    scored = []
    for study in studies:
        cp_info = compute_cp_relevance(study)
        if cp_info["is_cp_relevant"] and cp_info["cp_score"] >= min_cp_score:
            scored.append((study, cp_info))

    # Sort by CP score descending
    scored.sort(key=lambda x: x[1]["cp_score"], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# LLM prompt for CP Agent advisory
# ---------------------------------------------------------------------------
CP_ADVISOR_PROMPT = """\
You are a senior Clinical Pharmacology scientist advising a researcher who is \
looking for clinical trial protocols suitable for their CP/PK research.

Given the researcher's goal and a list of candidate studies (with CP relevance scores), \
provide a brief expert recommendation:

1. Which 3-5 studies are BEST for their specific research goal and WHY
2. What makes each recommended study ideal for CP analysis (PK data richness, \
   study design, dosing strategy, sampling scheme likelihood)
3. Any caveats (e.g., "likely redacted PK sections", "may lack individual PK data")
4. What the researcher should look for when they open the protocol

Focus on PRACTICAL advice a clinical pharmacologist would give. Be specific about \
what PK/PD content they'll likely find in each protocol.

Output format:
## Top Recommendations

### 1. [NCT ID] — [Drug Name]
**Why this is ideal:** [1-2 sentences]
**Expected CP content:** [what PK/PD sections to expect]
**Caveat:** [any warning]

### 2. ...

## General Advice
[1-2 sentences of practical guidance for this research goal]
"""
