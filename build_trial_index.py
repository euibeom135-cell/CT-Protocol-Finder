#!/usr/bin/env python3
"""One-time script to build a FAISS index of clinical trials for similarity search.

Fetches Phase 1 / Early Phase 1 trials with protocol documents from ClinicalTrials.gov,
embeds them using OpenAI embeddings, and saves a FAISS index + metadata to disk.

Usage:
    python3 build_trial_index.py                          # defaults: Phase 1, max 2000 trials
    python3 build_trial_index.py --max-trials 5000        # fetch more trials
    python3 build_trial_index.py --phases PHASE1 PHASE2   # custom phases
    python3 build_trial_index.py --all-phases              # all phases
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ctgov_client import CTGovClient
from similarity_engine import TrialSimilarityEngine


def fetch_trials(phases: list[str] | None, max_trials: int, require_docs: bool = True) -> list[dict]:
    """Fetch trials from ClinicalTrials.gov API."""
    client = CTGovClient()

    # Search queries designed to find CP-rich studies
    search_queries = [
        "pharmacokinetics",
        "first in human",
        "dose escalation",
        "bioavailability",
        "drug interaction",
        "food effect",
        "single ascending dose",
        "multiple ascending dose",
        "phase 1 healthy volunteers",
        "absorption distribution metabolism excretion",
        "bioequivalence",
        "dose finding",
        "safety tolerability pharmacokinetics",
    ]

    seen_ids: set[str] = set()
    all_trials: list[dict] = []

    for q in search_queries:
        if len(all_trials) >= max_trials:
            break

        print(f"  Searching: {q!r} ...")
        try:
            results, total = client.search_studies(
                query_term=q,
                phases=phases,
                page_size=100,
                max_pages=5,
            )
        except Exception as e:
            print(f"    Error: {e}")
            continue

        for r in results:
            if r.nct_id in seen_ids:
                continue

            # Only include trials with at least one document
            if require_docs and not any(
                d.has_protocol or d.has_sap or d.has_icf for d in r.documents
            ):
                continue

            seen_ids.add(r.nct_id)
            all_trials.append({
                "nct_id": r.nct_id,
                "brief_title": r.brief_title,
                "official_title": r.official_title,
                "brief_summary": r.brief_summary,
                "phases": r.phases,
                "enrollment": r.enrollment,
                "status": r.status,
                "conditions": r.conditions,
                "interventions": r.interventions,
                "sponsor": r.sponsor,
                "documents": [d.model_dump() for d in r.documents],
            })

            if len(all_trials) >= max_trials:
                break

        print(f"    Found {len(results)} results, {len(all_trials)} unique trials so far")
        time.sleep(0.3)  # Be nice to the API

    return all_trials


def main():
    parser = argparse.ArgumentParser(description="Build trial similarity index")
    parser.add_argument("--max-trials", type=int, default=2000, help="Max trials to index")
    parser.add_argument("--phases", nargs="+", default=["PHASE1", "EARLY_PHASE1"],
                        help="CT.gov phase filters")
    parser.add_argument("--all-phases", action="store_true", help="Include all phases")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--no-docs", action="store_true", help="Don't require documents")
    args = parser.parse_args()

    phases = None if args.all_phases else args.phases
    phase_str = "all phases" if args.all_phases else ", ".join(args.phases)

    print(f"\n{'='*60}")
    print(f"  Building Trial Similarity Index")
    print(f"  Phases: {phase_str}")
    print(f"  Max trials: {args.max_trials}")
    print(f"  Require documents: {not args.no_docs}")
    print(f"{'='*60}\n")

    # Step 1: Fetch trials
    print("Step 1/3: Fetching trials from ClinicalTrials.gov...")
    t0 = time.time()
    trials = fetch_trials(phases, args.max_trials, require_docs=not args.no_docs)
    print(f"\n  Fetched {len(trials)} trials in {time.time() - t0:.1f}s\n")

    if not trials:
        print("No trials found. Check your API connection and filters.")
        sys.exit(1)

    # Step 2: Build FAISS index
    print("Step 2/3: Embedding trials and building FAISS index...")
    t0 = time.time()
    engine = TrialSimilarityEngine()
    engine.build_index(trials)
    print(f"  Index built in {time.time() - t0:.1f}s\n")

    # Step 3: Save to disk
    print(f"Step 3/3: Saving to {args.output_dir}/...")
    engine.save(args.output_dir)

    # Print stats
    idx_size = (Path(args.output_dir) / "trial_index.faiss").stat().st_size / 1024 / 1024
    meta_size = (Path(args.output_dir) / "trial_metadata.json").stat().st_size / 1024 / 1024
    print(f"\n  Index file: {idx_size:.1f} MB")
    print(f"  Metadata file: {meta_size:.1f} MB")
    print(f"  Total trials indexed: {engine.total_indexed}")

    # Quick sanity check — search for the first trial
    print(f"\nSanity check: finding trials similar to {trials[0]['nct_id']}...")
    similar = engine.find_similar_by_nct(trials[0]["nct_id"], top_k=3)
    for s in similar:
        print(f"  {s['nct_id']} (similarity: {s['similarity_score']:.3f}) — {s['brief_title'][:80]}")

    print(f"\nDone! Index saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
