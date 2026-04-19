"""LLM engine — multi-provider support for query parsing and result ranking."""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from models import SearchIntent, StudyResult

load_dotenv()

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
PARSE_SYSTEM_PROMPT = """\
You are a clinical trial search assistant specializing in clinical pharmacology.

Given a researcher's description of what they need, extract structured search
parameters for ClinicalTrials.gov. Output ONLY valid JSON matching this schema:

{
  "search_terms": "free text keywords — drugs, diseases, modalities like 'small molecule' or 'biologics'",
  "conditions": ["specific disease/condition names ONLY — e.g. 'breast cancer', 'diabetes', 'NSCLC'"],
  "interventions": ["drug or treatment names — e.g. 'pembrolizumab', 'metformin'"],
  "study_types": ["clinical trial design concepts — see list below"],
  "phases": ["PHASE1", "PHASE2", "PHASE3", "PHASE4", "EARLY_PHASE1"],
  "statuses": ["COMPLETED", "RECRUITING", "ACTIVE_NOT_RECRUITING", "TERMINATED", "WITHDRAWN"],
  "require_protocol": true,
  "require_sap": false,
  "require_icf": false,
  "min_enrollment": null,
  "rationale": "brief explanation of your parameter choices"
}

Valid study_types (use these exact canonical names):
  "first in human", "single ascending dose", "multiple ascending dose",
  "dose escalation", "dose finding", "food effect", "drug-drug interaction",
  "pharmacokinetics", "pharmacodynamics", "bioequivalence",
  "relative bioavailability", "absolute bioavailability", "mass balance",
  "QTc", "thorough QT", "renal impairment", "hepatic impairment",
  "safety and tolerability", "immunogenicity", "pediatric", "Japanese bridging"

Guidelines:
- CRITICAL: study_types captures trial DESIGN concepts. "first in human" is a study type, NOT a condition. "dose escalation" is a study type, NOT an intervention. "pharmacokinetics" is a study type.
- CRITICAL: conditions must ONLY contain actual diseases (e.g. "breast cancer", "NSCLC"). NEVER put "small molecule", "biologics", "first in human", "dose escalation" in conditions.
- "small molecule", "biologics", "monoclonal antibody" are modalities — put them in search_terms.
- Prefer COMPLETED status when the user wants documents (completed trials more often upload docs)
- For "data-rich" requests, suggest min_enrollment of 100+
- If user mentions SAP or statistical analysis plan, set require_sap to true
- If user mentions ICF or informed consent, set require_icf to true
- require_protocol defaults to true unless the user only wants SAP/ICF
- Be generous with search terms — broader is better for finding documents
- CRITICAL: NO DUPLICATES across fields. If a drug name goes in "interventions", do NOT also put it in "search_terms". If a disease goes in "conditions", do NOT also put it in "search_terms". Each value should appear in exactly ONE field.
- search_terms should ONLY contain general modality keywords (e.g. "small molecule", "biologics") that don't fit in conditions, interventions, or study_types. If there's nothing extra, leave search_terms empty.
- Output ONLY the JSON object, no markdown, no explanation
"""

PARSE_OR_CLARIFY_SYSTEM_PROMPT = """\
You are a clinical trial search assistant specializing in clinical pharmacology.

Given a researcher's description of what they need, do ONE of:

1. If the query is clear enough to search, output structured search parameters.
2. If the query is too vague or ambiguous, ask 1-3 short clarifying questions.

Output ONLY valid JSON matching one of these two schemas:

Schema A (ready to search):
{
  "status": "ok",
  "intent": {
    "search_terms": "free text keywords — drugs, diseases, modalities like 'small molecule'",
    "conditions": ["specific disease/condition names ONLY — e.g. 'breast cancer', 'NSCLC'"],
    "interventions": ["drug or treatment names — e.g. 'pembrolizumab'"],
    "study_types": ["clinical trial design concepts — see valid list below"],
    "phases": ["PHASE1", "PHASE2", "PHASE3", "PHASE4", "EARLY_PHASE1"],
    "statuses": ["COMPLETED", "RECRUITING", "ACTIVE_NOT_RECRUITING", "TERMINATED", "WITHDRAWN"],
    "require_protocol": true,
    "require_sap": false,
    "require_icf": false,
    "min_enrollment": null,
    "rationale": "brief explanation of your parameter choices"
  },
  "confidence": 85,
  "reasoning": [
    "Classified 'tocilizumab' as intervention (it's a drug name, not a condition)",
    "Set Phase 2 based on explicit mention in query"
  ],
  "suggestions": [
    "Consider adding a study type like 'dose escalation' or 'pharmacokinetics' for more CP-relevant results",
    "No specific condition was mentioned — results will span all therapeutic areas"
  ]
}

IMPORTANT fields in Schema A:
- "confidence": integer 0-100 representing how confident you are in the parse. 90+ = very clear query, 60-89 = reasonable but some assumptions made, below 60 = significant guessing.
- "reasoning": array of 1-4 short strings explaining WHY you classified each term into its field. One line per classification decision.
- "suggestions": array of 0-3 short strings suggesting how to improve the query. Empty array if the query is already excellent. Focus on: missing study types, missing phases, overly broad searches, or CP-relevant additions.

Schema B (need clarification):
{
  "status": "clarify",
  "questions": ["What therapeutic area are you interested in?", "Do you need completed trials only?"],
  "rationale": "The query is too broad; narrowing the condition and status will yield better results."
}

Valid study_types (use these exact canonical names):
  "first in human", "single ascending dose", "multiple ascending dose",
  "dose escalation", "dose finding", "food effect", "drug-drug interaction",
  "pharmacokinetics", "pharmacodynamics", "bioequivalence",
  "relative bioavailability", "absolute bioavailability", "mass balance",
  "QTc", "thorough QT", "renal impairment", "hepatic impairment",
  "safety and tolerability", "immunogenicity", "pediatric", "Japanese bridging"

Guidelines:
- CRITICAL: study_types captures trial DESIGN concepts. Examples:
  * "first in human" / "FIH" → study_types (NOT conditions, NOT search_terms)
  * "dose escalation" / "dose finding" → study_types (NOT interventions)
  * "pharmacokinetics" / "PK" → study_types
  * "food effect" → study_types
  * "DDI" / "drug interaction" → study_types ["drug-drug interaction"]
  * "SAD" → study_types ["single ascending dose"]
  * "MAD" → study_types ["multiple ascending dose"]
- CRITICAL: conditions must ONLY contain actual diseases (e.g. "breast cancer", "diabetes"). NEVER put "small molecule", "biologics", "first in human", "dose escalation" in conditions.
- "small molecule", "biologics", "monoclonal antibody" are modalities → search_terms only.
- Ask for clarification when: the query has no specific disease/drug/phase/study type, is fewer than 5 words with no clear domain, or is extremely vague.
- Do NOT ask for clarification when: the user mentions a specific drug, disease, phase, or study type — even if brief. Err on the side of searching.
- Prefer COMPLETED status when the user wants documents.
- For "data-rich" requests, suggest min_enrollment of 100+.
- If user mentions SAP or statistical analysis plan, set require_sap to true.
- If user mentions ICF or informed consent, set require_icf to true.
- Be generous with search terms — broader is better for finding documents.
- CRITICAL: NO DUPLICATES across fields. If a drug name goes in "interventions", do NOT also put it in "search_terms". If a disease goes in "conditions", do NOT also put it in "search_terms". Each value should appear in exactly ONE field.
- search_terms should ONLY contain general modality keywords (e.g. "small molecule", "biologics") that don't fit in conditions, interventions, or study_types. If there's nothing extra, leave search_terms empty string "".
- Output ONLY the JSON object, no markdown, no explanation.
"""

RANK_SYSTEM_PROMPT = """\
You are a clinical pharmacology expert helping a researcher find the best
clinical trial protocols for their research.

Given the researcher's query and a list of clinical trial results, score each
study from 0-100 for relevance. Consider:
- Does it match the therapeutic area / condition?
- Does it match the requested STUDY TYPE (e.g., first in human, dose escalation, food effect, DDI)? Check the title AND summary for study design keywords.
- Does it have the right phase and enrollment size?
- Does it have the document types the researcher needs?
- Would this be a good example for clinical pharmacology research?

Study type matching is VERY important — if a researcher asks for "first in human" trials, prioritize studies whose titles/summaries mention FIH, first-in-human, SAD, MAD, dose escalation, etc.

Output ONLY a JSON array:
[{"nct_id": "NCT...", "score": 85, "explanation": "one sentence why"}]

Be generous with scores. Include ALL studies in your output.
"""

PREVIEW_SYSTEM_PROMPT = """\
You are a clinical pharmacology expert. Given the extracted text from a clinical
trial protocol, provide a brief 2-3 sentence summary of what clinical pharmacology
content is available. Mention the drug, indication, study phase, and note if you
see any major redactions that would limit its usefulness for CP review.
Keep it concise and practical.
"""

STRUCTURED_PREVIEW_PROMPT = """\
You are a senior clinical pharmacology scientist reviewing a protocol document.
The content below has been extracted with section structure and tables preserved.

Provide a structured assessment in this exact format:

**Drug & Indication**: [drug name, mechanism, therapeutic area]
**Study Design**: [phase, design type (FIH/SAD/MAD/crossover/etc), number of arms]
**Key PK Elements**: [PK sampling, bioanalytical methods, PK parameters planned]
**Dosing**: [dose levels, route, regimen]
**CP Relevance**: [High/Medium/Low] — [1 sentence why]
**Data Richness**: [what CP data this protocol will generate]
**Redactions/Limitations**: [any redacted sections or missing info]

Be specific — cite actual numbers (dose levels, timepoints, sample sizes) when visible in the text.
If a section is missing or not found, say "Not found in extracted content."
"""


class LLMEngine:
    """Multi-provider LLM engine (BullsAI, OpenAI, Anthropic)."""

    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.provider = provider or os.getenv("LLM_PROVIDER", "openai")
        self._setup(api_key, model)

    def _setup(self, api_key: str | None, model: str | None):
        if self.provider == "bullsai":
            import openai
            self.api_key = api_key or os.getenv("BULLSAI_API_KEY", "")
            self.model = model or os.getenv("BULLSAI_MODEL", "openai/gpt-oss-120b")
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=os.getenv("BULLSAI_BASE_URL", "https://gateway.bullsai.buffalo.edu/v1"),
            )
            self._call = self._call_openai_compat

        elif self.provider == "openai":
            import openai
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.cp_advisory_model = os.getenv("OPENAI_CP_ADVISORY_MODEL", self.model)
            self.client = openai.OpenAI(api_key=self.api_key)
            self._call = self._call_openai_compat

        elif self.provider == "anthropic":
            import anthropic
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
            self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self._call = self._call_anthropic

        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    # ------------------------------------------------------------------
    # Provider-specific call methods
    # ------------------------------------------------------------------
    def _call_openai_compat(self, system: str, user: str, max_tokens: int = 2000, model: str | None = None) -> str:
        use_model = model or self.model
        print(f"[LLM] calling {use_model} (max_tokens={max_tokens})")
        resp = self.client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""

    def _call_anthropic(self, system: str, user: str, max_tokens: int = 2000) -> str:
        resp = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.content[0].text

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def parse_query(self, user_query: str) -> SearchIntent:
        """Parse a natural language query into structured search parameters."""
        raw = self._call(PARSE_SYSTEM_PROMPT, user_query)
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        try:
            data = json.loads(raw)
            return SearchIntent(**data)
        except (json.JSONDecodeError, Exception):
            # Fallback: use the raw query as search terms
            return SearchIntent(
                search_terms=user_query,
                require_protocol=True,
                rationale="Could not parse query; using raw text as search terms.",
            )

    def parse_query_interactive(
        self, user_query: str, conversation: list[dict] | None = None
    ) -> dict:
        """Parse query with possible clarification loop. Returns dict with status field."""
        # Build user message including any prior clarification context
        if conversation and len(conversation) > 0:
            parts = [f"Original query: {user_query}"]
            for turn in conversation:
                parts.append(f"Assistant asked: {turn.get('question', '')}")
                parts.append(f"User answered: {turn.get('answer', '')}")
            # After 2 rounds of clarification, force a search intent
            if len(conversation) >= 2:
                parts.append(
                    "You have already asked for clarification twice. "
                    "You MUST now produce a search intent (status: ok). Do not ask more questions."
                )
            user_msg = "\n".join(parts)
        else:
            user_msg = user_query

        raw = self._call(PARSE_OR_CLARIFY_SYSTEM_PROMPT, user_msg)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        try:
            data = json.loads(raw)
            if data.get("status") == "clarify":
                return {
                    "status": "clarify",
                    "questions": data.get("questions", []),
                    "rationale": data.get("rationale", ""),
                }
            # status == "ok"
            intent_data = data.get("intent", data)
            intent = SearchIntent(**intent_data)
            return {
                "status": "ok",
                "intent": intent.model_dump(),
                "confidence": data.get("confidence"),
                "reasoning": data.get("reasoning", []),
                "suggestions": data.get("suggestions", []),
            }
        except (json.JSONDecodeError, Exception):
            # Fallback: return a default intent
            intent = SearchIntent(
                search_terms=user_query,
                require_protocol=True,
                rationale="Could not parse query; using raw text as search terms.",
            )
            return {"status": "ok", "intent": intent.model_dump()}

    def rank_results(
        self, user_query: str, results: list[StudyResult], top_n: int = 20
    ) -> list[StudyResult]:
        """Rank search results by relevance to the user's query."""
        if not results:
            return results

        # Build a compact summary of results for the LLM
        summaries = []
        for s in results[:top_n]:
            doc_types = []
            for d in s.documents:
                if d.has_protocol:
                    doc_types.append("Protocol")
                if d.has_sap:
                    doc_types.append("SAP")
                if d.has_icf:
                    doc_types.append("ICF")
            summaries.append({
                "nct_id": s.nct_id,
                "title": s.brief_title,
                "summary": (s.brief_summary or "")[:200],
                "phases": s.phases,
                "enrollment": s.enrollment,
                "conditions": s.conditions[:3],
                "interventions": s.interventions[:3],
                "documents": list(set(doc_types)),
            })

        user_msg = (
            f"Researcher's query: {user_query}\n\n"
            f"Studies to rank:\n{json.dumps(summaries, indent=2)}"
        )

        raw = self._call(RANK_SYSTEM_PROMPT, user_msg, max_tokens=3000)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        try:
            rankings = json.loads(raw)
            score_map = {r["nct_id"]: r for r in rankings}
            for s in results:
                if s.nct_id in score_map:
                    s.relevance_score = score_map[s.nct_id].get("score", 50)
                    s.relevance_explanation = score_map[s.nct_id].get("explanation", "")
            results.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        except (json.JSONDecodeError, Exception):
            pass  # Return unsorted if ranking fails

        return results

    def preview_protocol(self, pdf_text: str) -> str:
        """Generate a brief CP content summary from protocol text."""
        # Truncate to ~15k chars to stay within token limits
        truncated = pdf_text[:15000]
        return self._call(PREVIEW_SYSTEM_PROMPT, truncated, max_tokens=500)

    def preview_protocol_structured(self, sections_text: str) -> str:
        """Generate a detailed CP assessment from structured protocol content."""
        return self._call(STRUCTURED_PREVIEW_PROMPT, sections_text, max_tokens=1000)

    def cp_advise(self, system_prompt: str, user_msg: str) -> str:
        """CP Advisory call using the upgraded model (gpt-4o) for better reasoning."""
        model = getattr(self, "cp_advisory_model", None)
        print(f"[CP Advisory] using model: {model or self.model}")
        return self._call(system_prompt, user_msg, max_tokens=2000, model=model)
