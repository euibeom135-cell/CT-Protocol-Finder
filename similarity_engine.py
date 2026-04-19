"""Trial Similarity Search Engine using OpenAI embeddings + FAISS + Ontology.

Provides two entry points:
  1. find_similar_by_nct(nct_id)  — find trials similar to a known trial
  2. find_similar_by_text(text)   — find trials similar to free text (e.g., uploaded PDF)

Similarity is computed using a 4-layer composite score:
  - MeSH condition distance        (25%)
  - Drug class / mechanism match   (30%)
  - FIH taxonomy match             (30%)
  - Text embedding cosine sim      (15%)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv

from ontology_engine import OntologyEngine

load_dotenv()

# ---------------------------------------------------------------------------
# Embedding helper (OpenAI text-embedding-3-small — 1536 dims, ~$0.02/1M tokens)
# ---------------------------------------------------------------------------
_client = None

def _get_openai_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed a batch of texts using OpenAI API. Returns (N, 1536) float32 array."""
    client = _get_openai_client()
    # OpenAI allows up to 2048 texts per call; batch if needed
    all_vecs: list[list[float]] = []
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Truncate each text to ~8000 tokens (~32000 chars) to stay within limits
        batch = [t[:32000] for t in batch]
        resp = client.embeddings.create(input=batch, model=model)
        for item in resp.data:
            all_vecs.append(item.embedding)
    return np.array(all_vecs, dtype="float32")


def embed_single(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed a single text string. Returns (1536,) float32 array."""
    return embed_texts([text], model=model)[0]


# ---------------------------------------------------------------------------
# Trial text builder — combine metadata fields into a rich text for embedding
# ---------------------------------------------------------------------------
def trial_to_text(trial: dict) -> str:
    """Convert a trial metadata dict to a single text string for embedding."""
    parts = []
    if trial.get("brief_title"):
        parts.append(f"Title: {trial['brief_title']}")
    if trial.get("official_title"):
        parts.append(f"Official title: {trial['official_title']}")
    if trial.get("brief_summary"):
        parts.append(f"Summary: {trial['brief_summary']}")
    if trial.get("conditions"):
        conds = trial["conditions"] if isinstance(trial["conditions"], list) else [trial["conditions"]]
        parts.append(f"Conditions: {', '.join(conds)}")
    if trial.get("interventions"):
        intv = trial["interventions"] if isinstance(trial["interventions"], list) else [trial["interventions"]]
        parts.append(f"Interventions: {', '.join(intv)}")
    if trial.get("phases"):
        ph = trial["phases"] if isinstance(trial["phases"], list) else [trial["phases"]]
        parts.append(f"Phase: {', '.join(ph)}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------
class TrialSimilarityEngine:
    """FAISS-backed similarity search over clinical trials with ontology reranking."""

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None  # inner-product (cosine on L2-normed vecs)
        self.metadata: list[dict] = []  # parallel list of trial metadata dicts
        self.nct_to_idx: dict[str, int] = {}  # NCT ID → index position
        self.dim: int = 1536
        self.ontology_engine: OntologyEngine = OntologyEngine()

    # -- persistence ----------------------------------------------------------
    def save(self, directory: str) -> None:
        """Save FAISS index + metadata + ontology profiles to disk."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "trial_index.faiss"))
        with open(d / "trial_metadata.json", "w") as f:
            json.dump(self.metadata, f)
        # Save ontology profiles
        if self.ontology_engine.has_profiles:
            self.ontology_engine.save_profiles(directory)

    def load(self, directory: str) -> bool:
        """Load pre-built index from disk. Returns True if loaded successfully."""
        d = Path(directory)
        idx_path = d / "trial_index.faiss"
        meta_path = d / "trial_metadata.json"
        if not idx_path.exists() or not meta_path.exists():
            return False
        self.index = faiss.read_index(str(idx_path))
        with open(meta_path) as f:
            self.metadata = json.load(f)
        self.nct_to_idx = {m["nct_id"]: i for i, m in enumerate(self.metadata)}
        self.dim = self.index.d
        # Load or build ontology profiles
        if not self.ontology_engine.load_profiles(directory):
            print("[Ontology] No profiles file found — building from metadata...")
            self.ontology_engine.build_profiles_from_index(self.metadata)
            self.ontology_engine.save_profiles(directory)
        return True

    # -- index building -------------------------------------------------------
    def build_index(self, trials: list[dict], show_progress: bool = True) -> None:
        """Build FAISS index from a list of trial metadata dicts.

        Each dict should have at least: nct_id, brief_title, brief_summary.
        Optionally: official_title, conditions, interventions, phases, enrollment, status, sponsor, documents.
        """
        if not trials:
            raise ValueError("No trials to index")

        # Convert trials to text
        texts = [trial_to_text(t) for t in trials]

        # Embed in batches
        if show_progress:
            print(f"Embedding {len(texts)} trials...")

        vectors = embed_texts(texts)

        # L2-normalize for cosine similarity via inner product
        faiss.normalize_L2(vectors)

        # Build flat inner-product index (exact search; fast enough for <50k vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

        self.metadata = trials
        self.nct_to_idx = {t["nct_id"]: i for i, t in enumerate(trials)}
        self.dim = vectors.shape[1]

        # Build ontology profiles
        if show_progress:
            print("Building ontology profiles...")
        self.ontology_engine.build_profiles_from_index(trials)

        if show_progress:
            print(f"Index built: {self.index.ntotal} trials, {self.dim} dimensions")

    # -- search ---------------------------------------------------------------
    def find_similar_by_nct(self, nct_id: str, top_k: int = 15) -> list[dict]:
        """Find trials similar to a known trial by NCT ID."""
        if self.index is None:
            return []
        idx = self.nct_to_idx.get(nct_id)
        if idx is None:
            # Trial not in index — fetch its metadata from ClinicalTrials.gov and embed
            from ctgov_client import CTGovClient
            client = CTGovClient()
            results, _ = client.search_studies(query_term=nct_id, page_size=1, max_pages=1)
            if not results:
                return []
            trial = results[0]
            trial_dict = {
                "nct_id": trial.nct_id,
                "brief_title": trial.brief_title,
                "official_title": trial.official_title,
                "brief_summary": trial.brief_summary,
                "conditions": trial.conditions,
                "interventions": trial.interventions,
                "phases": trial.phases,
            }
            text = trial_to_text(trial_dict)
            query_profile = self.ontology_engine.build_profile(trial_dict)
            return self._search_by_text(text, top_k=top_k, exclude_nct=nct_id,
                                        query_profile=query_profile)

        # Build query profile from indexed metadata
        query_profile = self.ontology_engine.profiles.get(nct_id)
        if query_profile is None:
            query_profile = self.ontology_engine.build_profile(self.metadata[idx])

        # Reconstruct vector from index
        vec = self.index.reconstruct(idx).reshape(1, -1)
        return self._search_by_vector(vec, top_k=top_k + 1, exclude_nct=nct_id,
                                      query_profile=query_profile)

    def find_similar_by_text(self, text: str, top_k: int = 15,
                             conditions: list[str] | None = None,
                             interventions: list[str] | None = None) -> list[dict]:
        """Find trials similar to arbitrary text (e.g., from uploaded protocol).

        Args:
            text: The text to embed and search.
            top_k: Number of results to return.
            conditions: Optional condition hints for ontology matching.
            interventions: Optional intervention hints for ontology matching.
        """
        query_profile = self.ontology_engine.build_profile_from_text(
            text, conditions=conditions, interventions=interventions
        )
        return self._search_by_text(text, top_k=top_k, query_profile=query_profile)

    def _search_by_text(self, text: str, top_k: int = 15, exclude_nct: str = "",
                        query_profile=None) -> list[dict]:
        """Embed text and search the index."""
        if self.index is None:
            return []
        vec = embed_single(text).reshape(1, -1)
        faiss.normalize_L2(vec)
        return self._search_by_vector(vec, top_k=top_k, exclude_nct=exclude_nct,
                                      query_profile=query_profile)

    def _search_by_vector(self, vec: np.ndarray, top_k: int = 15,
                          exclude_nct: str = "", query_profile=None) -> list[dict]:
        """Search index by vector, then optionally rerank with ontology.

        Uses 3x oversampling when ontology reranking is active to allow
        semantically closer trials to surface even if they rank lower by
        pure text embedding.
        """
        # Fetch more candidates when doing ontology reranking
        has_ontology = query_profile is not None and self.ontology_engine.has_profiles
        fetch_k = min(top_k * 3 if has_ontology else top_k + 5, self.index.ntotal)

        scores, indices = self.index.search(vec, fetch_k)
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            if meta["nct_id"] == exclude_nct:
                continue
            result = {**meta, "similarity_score": float(score)}
            candidates.append(result)

        # Ontology reranking
        if has_ontology and query_profile is not None:
            candidates = self.ontology_engine.rerank_results(query_profile, candidates)

        return candidates[:top_k]

    @property
    def is_loaded(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    @property
    def total_indexed(self) -> int:
        return self.index.ntotal if self.index else 0
