"""3-Layer Ontology Similarity Engine for FIH/Dose-Escalation Clinical Trials.

Computes composite similarity scores using three layers:
  Layer 1: MeSH condition distance        (25% weight)
  Layer 2: Drug class / mechanism distance (30% weight)
  Layer 3: FIH taxonomy match             (30% weight)
  + Text embedding similarity             (15% weight, from FAISS)

Usage:
  engine = OntologyEngine()
  engine.build_profiles_from_index(trial_metadata_list)
  results = engine.rerank_results(query_profile, faiss_candidates)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ontology_data.mesh_subset import condition_to_mesh, condition_set_similarity
from ontology_data.drug_classes import drug_to_mechanism, intervention_set_similarity
from ontology_data.fih_taxonomy import (
    extract_fih_profile,
    fih_profile_similarity,
    fih_dimension_breakdown,
    FIHProfile,
)

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
WEIGHTS = {
    "mesh": 0.25,
    "drug_class": 0.30,
    "fih_taxonomy": 0.30,
    "embedding": 0.15,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class OntologyProfile:
    """Pre-computed ontology features for a trial."""
    nct_id: str
    mesh_codes: list[str] = field(default_factory=list)
    drug_classes: list[tuple[str, str] | None] = field(default_factory=list)
    fih_profile: FIHProfile = field(default_factory=FIHProfile)
    conditions: list[str] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "nct_id": self.nct_id,
            "mesh_codes": self.mesh_codes,
            "drug_classes": [list(dc) if dc else None for dc in self.drug_classes],
            "fih_profile": self.fih_profile.to_dict(),
            "conditions": self.conditions,
            "interventions": self.interventions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OntologyProfile:
        drug_classes = []
        for dc in d.get("drug_classes", []):
            if dc is not None:
                drug_classes.append(tuple(dc))
            else:
                drug_classes.append(None)

        return cls(
            nct_id=d["nct_id"],
            mesh_codes=d.get("mesh_codes", []),
            drug_classes=drug_classes,
            fih_profile=FIHProfile.from_dict(d.get("fih_profile", {})),
            conditions=d.get("conditions", []),
            interventions=d.get("interventions", []),
        )


@dataclass
class SimilarityBreakdown:
    """Detailed similarity score breakdown."""
    overall_score: float = 0.0
    mesh_score: float = 0.0
    drug_class_score: float = 0.0
    fih_score: float = 0.0
    embedding_score: float = 0.0
    mesh_detail: str = ""
    drug_class_detail: str = ""
    fih_detail: str = ""
    fih_sub_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 3),
            "mesh_score": round(self.mesh_score, 3),
            "drug_class_score": round(self.drug_class_score, 3),
            "fih_score": round(self.fih_score, 3),
            "embedding_score": round(self.embedding_score, 3),
            "mesh_detail": self.mesh_detail,
            "drug_class_detail": self.drug_class_detail,
            "fih_detail": self.fih_detail,
            "fih_sub_breakdown": {k: round(v, 3) for k, v in self.fih_sub_breakdown.items()},
        }


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------
class OntologyEngine:
    """Manages ontology profiles and computes multi-layer similarity."""

    def __init__(self):
        self.profiles: dict[str, OntologyProfile] = {}

    # -- Profile building -----------------------------------------------------
    def build_profile(self, trial: dict) -> OntologyProfile:
        """Build ontology profile from a trial metadata dict.

        Expected keys: nct_id, conditions (list), interventions (list),
                       brief_title, official_title, brief_summary.
        """
        nct_id = trial.get("nct_id", "")
        conditions = trial.get("conditions", [])
        interventions = trial.get("interventions", [])

        if isinstance(conditions, str):
            conditions = [conditions]
        if isinstance(interventions, str):
            interventions = [interventions]

        # Layer 1: MeSH codes
        mesh_codes = []
        for cond in conditions:
            code = condition_to_mesh(cond)
            if code:
                mesh_codes.append(code)

        # Layer 2: Drug classes
        drug_classes = []
        for intv in interventions:
            mech = drug_to_mechanism(intv)
            drug_classes.append(mech)  # None if unknown

        # Layer 3: FIH profile (from combined text)
        text_parts = []
        for key in ("brief_title", "official_title", "brief_summary"):
            val = trial.get(key, "")
            if val:
                text_parts.append(val)
        combined_text = " ".join(text_parts)
        fih_profile = extract_fih_profile(combined_text)

        return OntologyProfile(
            nct_id=nct_id,
            mesh_codes=mesh_codes,
            drug_classes=drug_classes,
            fih_profile=fih_profile,
            conditions=conditions,
            interventions=interventions,
        )

    def build_profile_from_text(
        self,
        text: str,
        conditions: list[str] | None = None,
        interventions: list[str] | None = None,
        nct_id: str = "QUERY",
    ) -> OntologyProfile:
        """Build profile from free text (for upload or ad-hoc query).

        conditions/interventions are optional hints; if not provided,
        only the FIH taxonomy layer is populated from text.
        """
        conditions = conditions or []
        interventions = interventions or []

        mesh_codes = [condition_to_mesh(c) for c in conditions if condition_to_mesh(c)]
        drug_classes = [drug_to_mechanism(i) for i in interventions]
        fih_profile = extract_fih_profile(text)

        return OntologyProfile(
            nct_id=nct_id,
            mesh_codes=mesh_codes,
            drug_classes=drug_classes,
            fih_profile=fih_profile,
            conditions=conditions,
            interventions=interventions,
        )

    def build_profiles_from_index(self, metadata: list[dict]) -> None:
        """Pre-compute ontology profiles for all indexed trials."""
        self.profiles.clear()
        for trial in metadata:
            profile = self.build_profile(trial)
            self.profiles[profile.nct_id] = profile
        print(f"[Ontology] Built profiles for {len(self.profiles)} trials")

    # -- Similarity computation -----------------------------------------------
    def compute_similarity(
        self,
        profile_a: OntologyProfile,
        profile_b: OntologyProfile,
        embedding_score: float = 0.0,
    ) -> SimilarityBreakdown:
        """Compute multi-layer similarity between two profiles.

        Args:
            profile_a: Query trial profile.
            profile_b: Candidate trial profile.
            embedding_score: FAISS cosine similarity (0-1).

        Returns:
            SimilarityBreakdown with overall and per-layer scores.
        """
        # Layer 1: MeSH condition similarity
        mesh_score, mesh_detail = condition_set_similarity(
            profile_a.conditions, profile_b.conditions
        )

        # Layer 2: Drug class similarity
        drug_score, drug_detail = intervention_set_similarity(
            profile_a.interventions, profile_b.interventions
        )

        # Layer 3: FIH taxonomy similarity
        fih_score = fih_profile_similarity(profile_a.fih_profile, profile_b.fih_profile)
        fih_sub = fih_dimension_breakdown(profile_a.fih_profile, profile_b.fih_profile)
        fih_detail = profile_b.fih_profile.summary()

        # Composite
        overall = (
            WEIGHTS["mesh"] * mesh_score
            + WEIGHTS["drug_class"] * drug_score
            + WEIGHTS["fih_taxonomy"] * fih_score
            + WEIGHTS["embedding"] * embedding_score
        )

        return SimilarityBreakdown(
            overall_score=overall,
            mesh_score=mesh_score,
            drug_class_score=drug_score,
            fih_score=fih_score,
            embedding_score=embedding_score,
            mesh_detail=mesh_detail,
            drug_class_detail=drug_detail,
            fih_detail=fih_detail,
            fih_sub_breakdown=fih_sub,
        )

    # -- Reranking ------------------------------------------------------------
    def rerank_results(
        self,
        query_profile: OntologyProfile,
        candidates: list[dict],
    ) -> list[dict]:
        """Rerank FAISS results using ontology scoring.

        Each candidate dict must have 'nct_id' and 'similarity_score' (from FAISS).
        Returns candidates with updated 'similarity_score' (composite) and
        added 'similarity_breakdown' dict.
        """
        reranked = []
        for cand in candidates:
            nct_id = cand.get("nct_id", "")
            embedding_score = cand.get("similarity_score", 0.0)

            # Get or build candidate profile
            cand_profile = self.profiles.get(nct_id)
            if cand_profile is None:
                cand_profile = self.build_profile(cand)

            # Compute multi-layer similarity
            breakdown = self.compute_similarity(
                query_profile, cand_profile, embedding_score
            )

            # Update candidate
            result = {**cand}
            result["similarity_score"] = breakdown.overall_score
            result["similarity_breakdown"] = breakdown.to_dict()
            reranked.append(result)

        # Sort by composite score descending
        reranked.sort(key=lambda x: x["similarity_score"], reverse=True)
        return reranked

    # -- Persistence ----------------------------------------------------------
    def save_profiles(self, directory: str) -> None:
        """Save pre-computed profiles to disk as JSON."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        profiles_data = {nct: p.to_dict() for nct, p in self.profiles.items()}
        with open(d / "ontology_profiles.json", "w") as f:
            json.dump(profiles_data, f, indent=2)
        print(f"[Ontology] Saved {len(profiles_data)} profiles to {d / 'ontology_profiles.json'}")

    def load_profiles(self, directory: str) -> bool:
        """Load pre-computed profiles from disk.

        Returns True if loaded successfully.
        """
        path = Path(directory) / "ontology_profiles.json"
        if not path.exists():
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self.profiles = {nct: OntologyProfile.from_dict(p) for nct, p in data.items()}
            print(f"[Ontology] Loaded {len(self.profiles)} profiles")
            return True
        except Exception as e:
            print(f"[Ontology] Failed to load profiles: {e}")
            return False

    @property
    def has_profiles(self) -> bool:
        return len(self.profiles) > 0
