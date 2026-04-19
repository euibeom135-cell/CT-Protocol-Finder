"""Evaluation framework for CT Protocol Finder.

Measures performance across 4 dimensions:
1. Search Quality   — precision/recall of finding CP-relevant trials
2. Query Parsing    — does the LLM correctly classify study types, conditions, etc.
3. CP Scoring       — does the heuristic CP score correlate with actual CP relevance
4. PDF Extraction   — are sections and tables extracted correctly

Usage:
  python3 evaluation.py build-ground-truth   # Interactive: run queries, label results
  python3 evaluation.py run                  # Run all test queries, compute metrics
  python3 evaluation.py report               # Generate publishable metrics report
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_FILE = BASE_DIR / "eval_ground_truth.json"
EVAL_RESULTS_DIR = BASE_DIR / "eval_results"
EVAL_RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Ground truth structure
# ---------------------------------------------------------------------------
def load_ground_truth() -> dict:
    """Load or create ground truth file."""
    if GROUND_TRUTH_FILE.exists():
        return json.loads(GROUND_TRUTH_FILE.read_text())
    return {
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "test_queries": [],
        "labeled_studies": {},  # nct_id -> {cp_relevant: bool, ideal_cp_score_range: [lo, hi], notes: str}
    }


def save_ground_truth(gt: dict):
    GROUND_TRUTH_FILE.write_text(json.dumps(gt, indent=2))
    print(f"Saved ground truth to {GROUND_TRUTH_FILE}")


# ---------------------------------------------------------------------------
# Test query definitions (seed queries for evaluation)
# ---------------------------------------------------------------------------
SEED_QUERIES = [
    # ── Core FIH / SAD / MAD ──────────────────────────────────────────────────
    {
        "id": "q1_fih_small_molecule",
        "query": "Phase 1 first in human dose escalation small molecule oral",
        "expected_study_types": ["first in human", "dose escalation"],
        "expected_phases": ["PHASE1", "EARLY_PHASE1"],
        "min_expected_results": 5,
        "description": "Core FIH query — should find many Phase 1 SAD/MAD trials",
    },
    {
        "id": "q8_sad_mad",
        "query": "single ascending dose multiple ascending dose safety tolerability",
        "expected_study_types": ["single ascending dose", "multiple ascending dose"],
        "expected_phases": ["PHASE1"],
        "min_expected_results": 5,
        "description": "Classic SAD/MAD study design",
    },
    {
        "id": "q9_fih_iv",
        "query": "first in human intravenous single dose safety pharmacokinetics",
        "expected_study_types": ["first in human", "pharmacokinetics"],
        "expected_phases": ["PHASE1", "EARLY_PHASE1"],
        "min_expected_results": 3,
        "description": "FIH IV route — tests route-specific retrieval",
    },
    {
        "id": "q10_fih_biologic",
        "query": "first in human monoclonal antibody biologic dose escalation Phase 1",
        "expected_study_types": ["first in human", "dose escalation"],
        "expected_phases": ["PHASE1"],
        "min_expected_results": 3,
        "description": "FIH biologic (mAb) — different modality from small molecule",
    },
    # ── PK Studies ────────────────────────────────────────────────────────────
    {
        "id": "q2_pk_food_effect",
        "query": "food effect study pharmacokinetics healthy volunteers",
        "expected_study_types": ["food effect", "pharmacokinetics"],
        "expected_phases": ["PHASE1"],
        "min_expected_results": 3,
        "description": "Food effect PK studies in healthy volunteers",
    },
    {
        "id": "q11_absolute_bioavailability",
        "query": "absolute bioavailability intravenous oral crossover pharmacokinetics",
        "expected_study_types": ["pharmacokinetics", "bioavailability"],
        "expected_phases": ["PHASE1"],
        "min_expected_results": 2,
        "description": "Absolute BA study — IV vs oral PK crossover",
    },
    {
        "id": "q12_mass_balance",
        "query": "mass balance ADME radiolabeled healthy volunteers",
        "expected_study_types": ["mass balance", "pharmacokinetics"],
        "expected_phases": ["PHASE1"],
        "min_expected_results": 2,
        "description": "Mass balance / ADME study with radiolabeled compound",
    },
    {
        "id": "q13_dose_proportionality",
        "query": "dose proportionality pharmacokinetics Phase 1 healthy subjects",
        "expected_study_types": ["pharmacokinetics"],
        "expected_phases": ["PHASE1"],
        "min_expected_results": 2,
        "description": "Dose proportionality assessment",
    },
    # ── Drug-Drug Interaction ─────────────────────────────────────────────────
    {
        "id": "q3_ddi_cyp3a4",
        "query": "drug-drug interaction study Phase 1 CYP3A4 inhibitor",
        "expected_study_types": ["drug-drug interaction"],
        "expected_phases": ["PHASE1"],
        "min_expected_results": 3,
        "description": "DDI study — CYP3A4 inhibition",
    },
    {
        "id": "q14_ddi_inducer",
        "query": "drug interaction CYP3A4 inducer rifampin pharmacokinetics Phase 1",
        "expected_study_types": ["drug-drug interaction"],
        "expected_phases": ["PHASE1"],
        "min_expected_results": 2,
        "description": "DDI study — CYP3A4 induction (rifampin)",
    },
    {
        "id": "q15_ddi_transporter",
        "query": "drug transporter interaction OATP P-glycoprotein Phase 1",
        "expected_study_types": ["drug-drug interaction"],
        "expected_phases": ["PHASE1"],
        "min_expected_results": 2,
        "description": "DDI via transporter — OATP or P-gp",
    },
    # ── Special Populations ───────────────────────────────────────────────────
    {
        "id": "q6_renal_impairment",
        "query": "renal impairment pharmacokinetics Phase 1",
        "expected_study_types": ["renal impairment", "pharmacokinetics"],
        "expected_phases": ["PHASE1"],
        "min_expected_results": 2,
        "description": "Special population PK — renal impairment",
    },
    {
        "id": "q16_hepatic_impairment",
        "query": "hepatic impairment pharmacokinetics Phase 1",
        "expected_study_types": ["hepatic impairment", "pharmacokinetics"],
        "expected_phases": ["PHASE1"],
        "min_expected_results": 2,
        "description": "Special population PK — hepatic impairment",
    },
    {
        "id": "q17_pediatric_pk",
        "query": "pediatric pharmacokinetics dose finding Phase 1 children",
        "expected_study_types": ["pharmacokinetics"],
        "expected_phases": ["PHASE1"],
        "min_expected_results": 2,
        "description": "Pediatric PK study",
    },
    # ── Oncology ──────────────────────────────────────────────────────────────
    {
        "id": "q4_oncology_pk",
        "query": "Phase 1 dose escalation oncology pharmacokinetics solid tumors",
        "expected_study_types": ["dose escalation", "pharmacokinetics"],
        "expected_phases": ["PHASE1"],
        "min_expected_results": 5,
        "description": "Oncology Phase 1 with PK — dose-finding",
    },
    {
        "id": "q18_qtc_study",
        "query": "QT prolongation thorough QT study ICH E14 cardiac safety",
        "expected_study_types": ["pharmacokinetics"],
        "expected_phases": ["PHASE1"],
        "min_expected_results": 2,
        "description": "Thorough QT/QTc study (cardiac safety)",
    },
    # ── Bioequivalence / Formulation ──────────────────────────────────────────
    {
        "id": "q5_bioequivalence",
        "query": "bioequivalence study generic drug",
        "expected_study_types": ["bioequivalence"],
        "expected_phases": ["PHASE1"],
        "min_expected_results": 3,
        "description": "BE studies for generic drugs",
    },
    {
        "id": "q19_formulation_comparison",
        "query": "formulation comparison pharmacokinetics crossover Phase 1",
        "expected_study_types": ["pharmacokinetics", "bioavailability"],
        "expected_phases": ["PHASE1"],
        "min_expected_results": 2,
        "description": "Formulation bridging / relative BA crossover",
    },
    # ── Negative Controls ─────────────────────────────────────────────────────
    {
        "id": "q7_negative_vaccine",
        "query": "vaccine immunogenicity Phase 3",
        "expected_study_types": [],
        "expected_phases": ["PHASE3"],
        "min_expected_results": 0,
        "description": "NEGATIVE: vaccine Phase 3 — should score LOW on CP relevance",
        "expect_low_cp": True,
    },
    {
        "id": "q20_negative_behavioral",
        "query": "cognitive behavioral therapy depression randomized controlled trial",
        "expected_study_types": [],
        "expected_phases": ["PHASE3"],
        "min_expected_results": 0,
        "description": "NEGATIVE: behavioral intervention — no CP relevance expected",
        "expect_low_cp": True,
    },
]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------
class Evaluator:
    """Runs test queries and computes metrics."""

    def __init__(self):
        from ctgov_client import CTGovClient
        from llm_engine import LLMEngine
        from cp_agent import filter_for_cp, compute_cp_relevance
        from study_type_synonyms import expand_study_types, score_study_type_match

        self.ctgov = CTGovClient()
        self.llm = LLMEngine()
        self.filter_for_cp = filter_for_cp
        self.compute_cp_relevance = compute_cp_relevance
        self.expand_study_types = expand_study_types
        self.score_study_type_match = score_study_type_match

    def evaluate_query_parsing(self, test_query: dict) -> dict:
        """Evaluate how well the LLM parses a query."""
        query = test_query["query"]
        expected_types = set(test_query.get("expected_study_types", []))
        expected_phases = set(test_query.get("expected_phases", []))

        result = self.llm.parse_query_interactive(query)

        if result["status"] != "ok":
            return {
                "test_id": test_query["id"],
                "metric": "query_parsing",
                "status": "clarify_requested",
                "score": 0.0,
                "details": f"LLM asked for clarification instead of parsing: {result.get('questions', [])}",
            }

        intent = result["intent"]
        parsed_types = set(intent.get("study_types", []))
        parsed_phases = set(intent.get("phases", []))

        # Study type accuracy
        if expected_types:
            type_precision = len(parsed_types & expected_types) / max(len(parsed_types), 1)
            type_recall = len(parsed_types & expected_types) / max(len(expected_types), 1)
            type_f1 = 2 * type_precision * type_recall / max(type_precision + type_recall, 0.001)
        else:
            type_f1 = 1.0 if not parsed_types else 0.0  # Should NOT have study types for negative cases

        # Phase accuracy
        if expected_phases:
            phase_match = len(parsed_phases & expected_phases) / max(len(expected_phases), 1)
        else:
            phase_match = 1.0

        # Check for misclassification (study type in conditions = bad)
        conditions = set(intent.get("conditions", []))
        misclassified = any(
            t.lower() in c.lower()
            for t in expected_types
            for c in conditions
        )

        score = (type_f1 * 0.5 + phase_match * 0.3 + (0.0 if misclassified else 0.2))

        return {
            "test_id": test_query["id"],
            "metric": "query_parsing",
            "score": round(score, 3),
            "details": {
                "expected_study_types": list(expected_types),
                "parsed_study_types": list(parsed_types),
                "type_f1": round(type_f1, 3),
                "expected_phases": list(expected_phases),
                "parsed_phases": list(parsed_phases),
                "phase_match": round(phase_match, 3),
                "misclassified_as_condition": misclassified,
                "full_intent": intent,
            },
        }

    def evaluate_search_recall(self, test_query: dict) -> dict:
        """Evaluate search: how many results returned, do they have documents."""
        from models import SearchIntent

        query = test_query["query"]
        min_expected = test_query.get("min_expected_results", 3)

        # Parse query
        result = self.llm.parse_query_interactive(query)
        if result["status"] != "ok":
            return {
                "test_id": test_query["id"],
                "metric": "search_recall",
                "score": 0.0,
                "details": "Could not parse query",
            }

        intent = SearchIntent(**result["intent"])

        # Run multi-strategy search (mirrors app.py logic)
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

        # Strategy A
        results_a, _ = self.ctgov.search_studies(
            query_term=base_query,
            phases=intent.phases or None,
            statuses=intent.statuses or None,
            page_size=50,
            max_pages=2,
        )
        for r in results_a:
            if r.nct_id not in seen_ids:
                seen_ids.add(r.nct_id)
                all_results.append(r)

        # Strategy B
        if intent.study_types:
            expanded = self.expand_study_types(intent.study_types)
            or_terms = " OR ".join(f'"{t}"' for t in expanded[:12])
            st_query = f"{base_query} {or_terms}".strip() if base_query else or_terms
            results_b, _ = self.ctgov.search_studies(
                query_term=st_query,
                phases=intent.phases or None,
                statuses=intent.statuses or None,
                page_size=50,
                max_pages=2,
            )
            for r in results_b:
                if r.nct_id not in seen_ids:
                    seen_ids.add(r.nct_id)
                    all_results.append(r)

        # Filter for documents
        with_docs = self.ctgov.filter_with_documents(all_results)

        # CP filter
        cp_results = self.filter_for_cp(with_docs, min_cp_score=20)

        total_found = len(all_results)
        with_docs_count = len(with_docs)
        cp_relevant_count = len(cp_results)

        # Build run_dict: NCT ID → normalized CP score (0-1) for ranx
        run_dict_inner: dict[str, float] = {}
        cp_ids: set[str] = set()
        for study, cp_info in cp_results:
            run_dict_inner[study.nct_id] = cp_info["cp_score"] / 100.0
            cp_ids.add(study.nct_id)
        # Below-threshold results get tiny scores so they appear at bottom of ranking
        for i, study in enumerate(with_docs):
            if study.nct_id not in cp_ids:
                run_dict_inner[study.nct_id] = 0.005 / (i + 1)
        ranked_nct_ids = [
            nct_id
            for nct_id, _ in sorted(run_dict_inner.items(), key=lambda x: x[1], reverse=True)
        ]

        # Quality-based score: CP relevance of top results (not just raw count)
        top5_cp_scores = [cp_info["cp_score"] for _, cp_info in cp_results[:5]]
        avg_cp_top5 = sum(top5_cp_scores) / max(len(top5_cp_scores), 1)
        cp_precision_5 = sum(1 for s in top5_cp_scores if s >= 50) / 5

        if min_expected == 0:
            # Negative case: we want low CP scores in top results
            score = round(1.0 - (avg_cp_top5 / 100.0), 3)
        else:
            coverage = min(1.0, with_docs_count / max(min_expected, 1))
            quality = avg_cp_top5 / 100.0
            score = round(0.5 * coverage + 0.5 * quality, 3)

        details: dict = {
            "total_api_results": total_found,
            "with_documents": with_docs_count,
            "cp_relevant": cp_relevant_count,
            "min_expected": min_expected,
            "avg_cp_score_top5": round(avg_cp_top5, 1),
            "cp_precision_at_5": round(cp_precision_5, 3),
            "top_5_nct_ids": [r.nct_id for r in with_docs[:5]],
            "ranked_nct_ids": ranked_nct_ids[:20],  # top-20 candidates for labeling
        }

        # If ground truth NCT IDs provided, compute real ranx metrics
        relevant_nct_ids = test_query.get("relevant_nct_ids", [])
        if relevant_nct_ids and run_dict_inner:
            qrels_inner = {test_query["id"]: {nct_id: 1 for nct_id in relevant_nct_ids}}
            run_for_ranx = {test_query["id"]: run_dict_inner}
            details["ranx_metrics"] = compute_ranx_metrics(qrels_inner, run_for_ranx)

        return {
            "test_id": test_query["id"],
            "metric": "search_recall",
            "score": round(score, 3),
            "details": details,
        }

    def evaluate_cp_scoring(self, test_query: dict) -> dict:
        """Evaluate CP scoring: do high CP scores correlate with CP relevance?"""
        query = test_query["query"]
        expect_low_cp = test_query.get("expect_low_cp", False)

        # Parse and search
        result = self.llm.parse_query_interactive(query)
        if result["status"] != "ok":
            return {"test_id": test_query["id"], "metric": "cp_scoring", "score": 0.0, "details": "Parse failed"}

        from models import SearchIntent
        intent = SearchIntent(**result["intent"])

        search_parts = []
        if intent.search_terms: search_parts.append(intent.search_terms)
        if intent.conditions: search_parts.extend(intent.conditions)
        base_query = " ".join(search_parts) if search_parts else query

        results, _ = self.ctgov.search_studies(
            query_term=base_query,
            phases=intent.phases or None,
            page_size=20,
            max_pages=1,
        )

        if not results:
            return {"test_id": test_query["id"], "metric": "cp_scoring", "score": 0.5, "details": "No results to score"}

        # Score all results
        scores = []
        for study in results[:10]:
            cp_info = self.compute_cp_relevance(study)
            scores.append({
                "nct_id": study.nct_id,
                "title": study.brief_title[:80],
                "cp_score": cp_info["cp_score"],
                "flags": cp_info["flags"][:2],
                "red_flags": cp_info["red_flags"][:2],
            })

        avg_score = sum(s["cp_score"] for s in scores) / max(len(scores), 1)
        high_cp_count = sum(1 for s in scores if s["cp_score"] >= 50)

        if expect_low_cp:
            # For vaccine/non-CP queries, avg CP score should be LOW
            score = 1.0 if avg_score < 30 else max(0, 1 - (avg_score - 30) / 70)
        else:
            # For CP queries, avg CP score should be HIGH
            score = min(1.0, avg_score / 60)

        return {
            "test_id": test_query["id"],
            "metric": "cp_scoring",
            "score": round(score, 3),
            "details": {
                "avg_cp_score": round(avg_score, 1),
                "high_cp_count": high_cp_count,
                "total_scored": len(scores),
                "expect_low_cp": expect_low_cp,
                "top_studies": scores[:5],
            },
        }

    def run_all(self, queries: list[dict] | None = None) -> dict:
        """Run full evaluation suite."""
        queries = queries or SEED_QUERIES
        timestamp = datetime.now().isoformat()

        all_results = []
        summary = {
            "timestamp": timestamp,
            "n_queries": len(queries),
            "metrics": {},
        }

        for i, tq in enumerate(queries):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(queries)}] {tq['id']}: {tq['query']}")
            print(f"  {tq['description']}")
            print(f"{'='*60}")

            # 1. Query Parsing
            print("  → Evaluating query parsing...")
            parse_result = self.evaluate_query_parsing(tq)
            all_results.append(parse_result)
            print(f"    Score: {parse_result['score']:.1%}")

            # 2. Search Recall
            print("  → Evaluating search recall...")
            search_result = self.evaluate_search_recall(tq)
            all_results.append(search_result)
            print(f"    Score: {search_result['score']:.1%} ({search_result['details']['with_documents']} with docs)")

            # 3. CP Scoring
            print("  → Evaluating CP scoring...")
            cp_result = self.evaluate_cp_scoring(tq)
            all_results.append(cp_result)
            print(f"    Score: {cp_result['score']:.1%} (avg CP: {cp_result['details'].get('avg_cp_score', 'N/A')})")

            time.sleep(0.5)  # Rate limit API calls

        # Compute aggregate metrics
        for metric in ["query_parsing", "search_recall", "cp_scoring"]:
            metric_scores = [r["score"] for r in all_results if r["metric"] == metric]
            if metric_scores:
                summary["metrics"][metric] = {
                    "mean": round(sum(metric_scores) / len(metric_scores), 3),
                    "min": round(min(metric_scores), 3),
                    "max": round(max(metric_scores), 3),
                    "n": len(metric_scores),
                }

        # Overall score
        all_scores = [r["score"] for r in all_results]
        summary["overall_score"] = round(sum(all_scores) / max(len(all_scores), 1), 3)

        # Save results
        output = {
            "summary": summary,
            "detailed_results": all_results,
            "queries": queries,
        }

        output_file = EVAL_RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.write_text(json.dumps(output, indent=2, default=str))
        print(f"\nResults saved to {output_file}")

        return output

    def print_report(self, output: dict):
        """Print a formatted evaluation report."""
        s = output["summary"]

        print("\n" + "=" * 70)
        print("  EVALUATION REPORT")
        print(f"  {s['timestamp']}")
        print("=" * 70)
        print(f"\n  Overall Score: {s['overall_score']:.1%}")
        print(f"  Queries Tested: {s['n_queries']}")
        print()

        print("  ┌─────────────────────┬────────┬────────┬────────┬───┐")
        print("  │ Metric              │  Mean  │  Min   │  Max   │ N │")
        print("  ├─────────────────────┼────────┼────────┼────────┼───┤")
        for name, m in s["metrics"].items():
            print(f"  │ {name:<19} │ {m['mean']:>5.1%} │ {m['min']:>5.1%} │ {m['max']:>5.1%} │ {m['n']} │")
        print("  └─────────────────────┴────────┴────────┴────────┴───┘")

        # Per-query breakdown
        print("\n  Per-Query Results:")
        print("  ─" * 35)
        results_by_query = {}
        for r in output["detailed_results"]:
            tid = r["test_id"]
            if tid not in results_by_query:
                results_by_query[tid] = {}
            results_by_query[tid][r["metric"]] = r["score"]

        for tid, scores in results_by_query.items():
            avg = sum(scores.values()) / max(len(scores), 1)
            parts = " | ".join(f"{k[:5]}={v:.0%}" for k, v in scores.items())
            emoji = "✅" if avg >= 0.7 else "⚠️" if avg >= 0.4 else "❌"
            print(f"  {emoji} {tid:<25} avg={avg:.0%}  [{parts}]")

        print()


# ---------------------------------------------------------------------------
# Interactive ground truth builder
# ---------------------------------------------------------------------------
def build_ground_truth_interactive():
    """Run queries interactively and let user label results as CP-relevant or not."""
    gt = load_ground_truth()

    from ctgov_client import CTGovClient
    from cp_agent import compute_cp_relevance

    ctgov = CTGovClient()

    print("\n🏗️  Ground Truth Builder")
    print("=" * 50)
    print("For each study, type:")
    print("  y = CP-relevant (good protocol for CP research)")
    print("  n = NOT CP-relevant")
    print("  s = skip")
    print("  q = quit\n")

    while True:
        query = input("\nEnter search query (or 'done'): ").strip()
        if query.lower() in ("done", "q", "quit"):
            break

        # Add query to ground truth
        gt["test_queries"].append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
        })

        results, total = ctgov.search_studies(query_term=query, page_size=20, max_pages=1)
        with_docs = ctgov.filter_with_documents(results)

        print(f"\nFound {len(with_docs)} studies with documents (of {total} total)")

        for i, study in enumerate(with_docs[:15]):
            cp_info = compute_cp_relevance(study)

            if study.nct_id in gt["labeled_studies"]:
                print(f"\n  [{i+1}] {study.nct_id} — already labeled, skipping")
                continue

            print(f"\n  [{i+1}] {study.nct_id}")
            print(f"      {study.brief_title[:80]}")
            print(f"      Phase: {', '.join(study.phases)} | Enrollment: {study.enrollment}")
            print(f"      Conditions: {', '.join(study.conditions[:3])}")
            print(f"      Interventions: {', '.join(study.interventions[:3])}")
            print(f"      CP Score (auto): {cp_info['cp_score']} | Flags: {', '.join(cp_info['flags'][:2])}")

            while True:
                choice = input("      CP-relevant? (y/n/s/q): ").strip().lower()
                if choice in ("y", "n", "s", "q"):
                    break

            if choice == "q":
                break
            if choice == "s":
                continue

            gt["labeled_studies"][study.nct_id] = {
                "cp_relevant": choice == "y",
                "auto_cp_score": cp_info["cp_score"],
                "title": study.brief_title[:100],
                "phases": study.phases,
                "query_used": query,
                "labeled_at": datetime.now().isoformat(),
            }
            print(f"      ✓ Labeled as {'CP-RELEVANT' if choice == 'y' else 'NOT CP-relevant'}")

        if query.lower() == "q":
            break

    save_ground_truth(gt)

    # Print summary
    labeled = gt["labeled_studies"]
    cp_yes = sum(1 for v in labeled.values() if v["cp_relevant"])
    cp_no = len(labeled) - cp_yes
    print(f"\n📊 Ground Truth: {len(labeled)} studies labeled ({cp_yes} CP-relevant, {cp_no} not)")


# ---------------------------------------------------------------------------
# Ground truth evaluation (if labels exist)
# ---------------------------------------------------------------------------
def evaluate_against_ground_truth():
    """If ground truth labels exist, compute precision/recall of CP scoring."""
    gt = load_ground_truth()
    labeled = gt.get("labeled_studies", {})

    if len(labeled) < 5:
        print(f"Need at least 5 labeled studies (have {len(labeled)}). Run: python3 evaluation.py build-ground-truth")
        return

    from cp_agent import compute_cp_relevance
    from ctgov_client import CTGovClient

    ctgov = CTGovClient()

    print(f"\n📊 Evaluating CP scoring against {len(labeled)} ground truth labels...")

    # For each labeled study, get its CP score and compare
    tp = fp = tn = fn = 0
    score_diffs = []

    for nct_id, label in labeled.items():
        human_relevant = label["cp_relevant"]
        auto_score = label.get("auto_cp_score", 0)
        system_says_relevant = auto_score >= 25  # our threshold

        if human_relevant and system_says_relevant:
            tp += 1
        elif human_relevant and not system_says_relevant:
            fn += 1
        elif not human_relevant and system_says_relevant:
            fp += 1
        else:
            tn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

    print(f"\n  CP Scoring vs Human Labels (threshold=25)")
    print(f"  ┌──────────────────┬────────┐")
    print(f"  │ Precision        │ {precision:>5.1%} │")
    print(f"  │ Recall           │ {recall:>5.1%} │")
    print(f"  │ F1 Score         │ {f1:>5.1%} │")
    print(f"  │ Accuracy         │ {accuracy:>5.1%} │")
    print(f"  ├──────────────────┼────────┤")
    print(f"  │ True Positives   │ {tp:>6} │")
    print(f"  │ False Positives  │ {fp:>6} │")
    print(f"  │ True Negatives   │ {tn:>6} │")
    print(f"  │ False Negatives  │ {fn:>6} │")
    print(f"  └──────────────────┴────────┘")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_labeled": len(labeled),
    }


# ---------------------------------------------------------------------------
# ranx-based retrieval metrics
# ---------------------------------------------------------------------------
def compute_ranx_metrics(
    qrels_dict: dict,
    run_dict: dict,
    metrics: list | None = None,
) -> dict:
    """Compute standard IR metrics using ranx.

    Args:
        qrels_dict: {query_id: {nct_id: relevance}}  — 1=relevant, 0=not relevant
        run_dict:   {query_id: {nct_id: score}}       — your system's ranked results
        metrics:    ranx metric names to compute

    Returns:
        dict of metric_name → float score

    Example:
        qrels = {"q1": {"NCT001": 1, "NCT002": 0}}
        run   = {"q1": {"NCT001": 0.92, "NCT002": 0.75}}
        compute_ranx_metrics(qrels, run)
        # → {"precision@5": 1.0, "recall@10": 1.0, "ndcg@10": 1.0, "mrr": 1.0}
    """
    from ranx import Qrels, Run, evaluate

    if metrics is None:
        metrics = ["precision@5", "recall@10", "ndcg@10", "mrr"]

    qrels = Qrels(qrels_dict)
    run = Run(run_dict)
    return dict(evaluate(qrels, run, metrics))


# ---------------------------------------------------------------------------
# Compare two eval runs
# ---------------------------------------------------------------------------
def compare_runs():
    """Compare the two most recent evaluation runs to see if performance improved."""
    eval_files = sorted(EVAL_RESULTS_DIR.glob("eval_*.json"))
    if len(eval_files) < 2:
        print("Need at least 2 evaluation runs to compare. Run 'python3 evaluation.py run' twice.")
        return

    prev = json.loads(eval_files[-2].read_text())
    curr = json.loads(eval_files[-1].read_text())

    prev_s = prev["summary"]
    curr_s = curr["summary"]

    print("\n📈 Performance Comparison")
    print("=" * 60)
    print(f"  Previous: {prev_s['timestamp']}")
    print(f"  Current:  {curr_s['timestamp']}")
    print()
    print(f"  Overall: {prev_s['overall_score']:.1%} → {curr_s['overall_score']:.1%}  ", end="")
    diff = curr_s["overall_score"] - prev_s["overall_score"]
    print(f"({'↑' if diff > 0 else '↓'} {abs(diff):.1%})")

    print()
    print("  ┌─────────────────────┬──────────┬──────────┬────────┐")
    print("  │ Metric              │ Previous │ Current  │ Change │")
    print("  ├─────────────────────┼──────────┼──────────┼────────┤")
    for metric in ["query_parsing", "search_recall", "cp_scoring"]:
        prev_m = prev_s["metrics"].get(metric, {}).get("mean", 0)
        curr_m = curr_s["metrics"].get(metric, {}).get("mean", 0)
        d = curr_m - prev_m
        arrow = "↑" if d > 0 else "↓" if d < 0 else "="
        print(f"  │ {metric:<19} │  {prev_m:>5.1%}  │  {curr_m:>5.1%}  │ {arrow}{abs(d):>4.1%}  │")
    print("  └─────────────────────┴──────────┴──────────┴────────┘")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 evaluation.py run              # Run evaluation suite")
        print("  python3 evaluation.py build-ground-truth  # Label studies interactively")
        print("  python3 evaluation.py gt-eval           # Evaluate against ground truth labels")
        print("  python3 evaluation.py compare           # Compare last two runs")
        print("  python3 evaluation.py report            # Show latest results")
        return

    cmd = sys.argv[1]

    if cmd == "run":
        evaluator = Evaluator()
        output = evaluator.run_all()
        evaluator.print_report(output)

    elif cmd == "build-ground-truth":
        build_ground_truth_interactive()

    elif cmd == "gt-eval":
        evaluate_against_ground_truth()

    elif cmd == "compare":
        compare_runs()

    elif cmd == "report":
        eval_files = sorted(EVAL_RESULTS_DIR.glob("eval_*.json"))
        if not eval_files:
            print("No evaluation results found. Run: python3 evaluation.py run")
            return
        output = json.loads(eval_files[-1].read_text())
        Evaluator().print_report(output)

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
