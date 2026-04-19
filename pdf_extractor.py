"""Smart PDF extraction using docling — structured sections, tables, and markdown."""
from __future__ import annotations

import os
import re
from pathlib import Path

from models import ProtocolContent

# ---------------------------------------------------------------------------
# Docling-based extraction (primary)
# ---------------------------------------------------------------------------

def extract_protocol(pdf_path: str) -> ProtocolContent:
    """Extract structured content from a protocol PDF.

    Uses docling for high-quality extraction with table preservation.
    Falls back to pdfplumber if docling fails.
    """
    pdf_path = str(pdf_path)

    try:
        return _extract_with_docling(pdf_path)
    except Exception as e:
        print(f"[PDF] docling failed ({e}), falling back to pdfplumber")
        return _extract_with_pdfplumber(pdf_path)


def _extract_with_docling(pdf_path: str) -> ProtocolContent:
    """Primary extraction using docling."""
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
    )
    from docling.document_converter import PdfFormatOption

    # Configure for accurate table extraction
    pipeline_options = PdfPipelineOptions(do_table_structure=True)
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.table_structure_options.do_cell_matching = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print(f"[PDF] Extracting with docling: {Path(pdf_path).name}")
    result = converter.convert(pdf_path)
    doc = result.document

    # Get full markdown
    full_markdown = doc.export_to_markdown()

    # Get page count
    page_count = len(doc.pages) if doc.pages else 0

    # Extract tables as markdown
    tables = []
    for table in doc.tables:
        try:
            md = table.export_to_markdown(doc=doc)
            if md and md.strip():
                tables.append(md.strip())
        except Exception:
            pass

    # Extract sections by heading
    sections = _extract_sections_from_markdown(full_markdown)

    # Check for redactions (common in clinical trial protocols)
    has_redactions = _check_redactions(full_markdown)

    print(f"[PDF] Extracted: {page_count} pages, {len(sections)} sections, {len(tables)} tables")

    return ProtocolContent(
        full_text=full_markdown,
        sections=sections,
        tables=tables,
        page_count=page_count,
        has_redactions=has_redactions,
        source_file=pdf_path,
    )


def _extract_with_pdfplumber(pdf_path: str) -> ProtocolContent:
    """Fallback extraction using pdfplumber."""
    import pdfplumber

    text = ""
    page_count = 0

    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages[:80]:  # First 80 pages
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    sections = _extract_sections_from_text(text)
    has_redactions = _check_redactions(text)

    return ProtocolContent(
        full_text=text,
        sections=sections,
        tables=[],  # pdfplumber doesn't extract tables well as markdown
        page_count=page_count,
        has_redactions=has_redactions,
        source_file=pdf_path,
    )


# ---------------------------------------------------------------------------
# Section extraction helpers
# ---------------------------------------------------------------------------

# Protocol section patterns (common in clinical trial protocols)
CP_SECTION_PATTERNS = [
    # Key CP sections
    (r"study\s*design", "Study Design"),
    (r"pharmacokinet", "Pharmacokinetics"),
    (r"pharmacodynam", "Pharmacodynamics"),
    (r"dose\s*escalat|dosing|dose\s*select|dose\s*level", "Dosing"),
    (r"pk\s*sampl|blood\s*sampl|sample\s*collect|sampling\s*schedule", "PK Sampling"),
    (r"bioanalytical|analytical\s*method|assay", "Bioanalytical Methods"),
    (r"statistic|statistical\s*analysis|sample\s*size", "Statistical Analysis"),
    (r"primary\s*(endpoint|objective|outcome)", "Primary Endpoints"),
    (r"secondary\s*(endpoint|objective|outcome)", "Secondary Endpoints"),
    (r"eligib|inclusion\s*criteria|exclusion\s*criteria", "Eligibility Criteria"),
    (r"safety|adverse\s*event|tolerability", "Safety"),
    (r"drug.drug\s*interact|DDI", "Drug-Drug Interaction"),
    (r"food\s*effect|fed.*fast|fasting", "Food Effect"),
    # General sections
    (r"introduction|background", "Introduction"),
    (r"study\s*objective|objective", "Objectives"),
    (r"study\s*population|subject|participant", "Study Population"),
    (r"study\s*procedure|procedure", "Study Procedures"),
    (r"investigational\s*product|study\s*drug|study\s*treatment", "Study Drug"),
    (r"concomitant\s*medic", "Concomitant Medications"),
    (r"discontinu|withdraw", "Discontinuation"),
    (r"data\s*management|data\s*handling", "Data Management"),
    (r"ethic|irb|informed\s*consent", "Ethics"),
    (r"reference|bibliograph", "References"),
    (r"abbreviat|glossar", "Abbreviations"),
    (r"synopsis|summary", "Synopsis"),
    (r"schedule\s*of\s*(assess|event|activit)", "Schedule of Assessments"),
]


def _extract_sections_from_markdown(markdown: str) -> dict[str, str]:
    """Extract named sections from markdown using heading patterns."""
    sections: dict[str, str] = {}

    # Split by markdown headings (## or ###)
    lines = markdown.split("\n")
    current_heading = ""
    current_text: list[str] = []

    for line in lines:
        # Check if this is a heading (## or ### level)
        heading_match = re.match(r"^#{1,4}\s+(.+)$", line.strip())
        if heading_match:
            # Save previous section
            if current_heading and current_text:
                section_name = _classify_section(current_heading)
                text = "\n".join(current_text).strip()
                if section_name and text:
                    # Append if section already exists (e.g., multiple PK sections)
                    if section_name in sections:
                        sections[section_name] += "\n\n" + text
                    else:
                        sections[section_name] = text

            current_heading = heading_match.group(1).strip()
            current_text = []
        else:
            current_text.append(line)

    # Don't forget last section
    if current_heading and current_text:
        section_name = _classify_section(current_heading)
        text = "\n".join(current_text).strip()
        if section_name and text:
            if section_name in sections:
                sections[section_name] += "\n\n" + text
            else:
                sections[section_name] = text

    return sections


def _extract_sections_from_text(text: str) -> dict[str, str]:
    """Extract sections from plain text (pdfplumber fallback)."""
    sections: dict[str, str] = {}

    # Look for numbered section headers (e.g., "1. INTRODUCTION", "5.1 Study Design")
    lines = text.split("\n")
    current_heading = ""
    current_text: list[str] = []

    for line in lines:
        # Check for section header patterns
        header_match = re.match(
            r"^\s*(\d+\.?\d*\.?\d*)\s+([A-Z][A-Z\s&/,()-]{2,})$",
            line.strip()
        )
        if header_match:
            if current_heading and current_text:
                section_name = _classify_section(current_heading)
                text_block = "\n".join(current_text).strip()
                if section_name and text_block:
                    if section_name in sections:
                        sections[section_name] += "\n\n" + text_block
                    else:
                        sections[section_name] = text_block

            current_heading = header_match.group(2).strip()
            current_text = []
        else:
            current_text.append(line)

    if current_heading and current_text:
        section_name = _classify_section(current_heading)
        text_block = "\n".join(current_text).strip()
        if section_name and text_block:
            if section_name in sections:
                sections[section_name] += "\n\n" + text_block
            else:
                sections[section_name] = text_block

    return sections


def _classify_section(heading: str) -> str:
    """Map a heading to a known section name, or return empty string."""
    heading_lower = heading.lower().strip()

    for pattern, name in CP_SECTION_PATTERNS:
        if re.search(pattern, heading_lower):
            return name

    # If heading is long enough and not classified, use it as-is (cleaned up)
    if len(heading_lower) > 3:
        # Capitalize nicely
        return heading.strip().title()[:60]

    return ""


def _check_redactions(text: str) -> bool:
    """Check if the document contains redacted content."""
    redaction_patterns = [
        r"\[redact",
        r"\[REDACT",
        r"CONFIDENTIAL.*REDACT",
        r"█+",  # Black bar characters
        r"\*{5,}",  # Multiple asterisks used as redaction
        r"XXXXX",
        r"\[.*removed.*\]",
        r"\[.*withheld.*\]",
        r"CCI",  # Confidential Commercial Information
    ]

    for pattern in redaction_patterns:
        if re.search(pattern, text[:50000]):  # Check first 50k chars
            return True

    return False


# ---------------------------------------------------------------------------
# Section summary for LLM
# ---------------------------------------------------------------------------

def format_sections_for_llm(content: ProtocolContent, max_chars: int = 20000) -> str:
    """Format extracted sections into a structured prompt for LLM analysis.

    Prioritizes CP-relevant sections and includes tables.
    """
    parts: list[str] = []
    char_count = 0

    # Priority order for CP analysis
    priority_sections = [
        "Synopsis", "Study Design", "Objectives", "Primary Endpoints",
        "Dosing", "Pharmacokinetics", "PK Sampling", "Pharmacodynamics",
        "Schedule of Assessments", "Bioanalytical Methods",
        "Statistical Analysis", "Study Population", "Eligibility Criteria",
        "Safety", "Drug-Drug Interaction", "Food Effect",
        "Study Drug", "Secondary Endpoints",
    ]

    # Add priority sections first
    for section_name in priority_sections:
        if section_name in content.sections:
            text = content.sections[section_name]
            # Truncate individual sections if too long
            if len(text) > 3000:
                text = text[:3000] + "\n... [truncated]"
            entry = f"\n### {section_name}\n{text}\n"
            if char_count + len(entry) > max_chars:
                break
            parts.append(entry)
            char_count += len(entry)

    # Add remaining sections
    for section_name, text in content.sections.items():
        if section_name in priority_sections:
            continue
        if len(text) > 2000:
            text = text[:2000] + "\n... [truncated]"
        entry = f"\n### {section_name}\n{text}\n"
        if char_count + len(entry) > max_chars:
            break
        parts.append(entry)
        char_count += len(entry)

    # Add tables at the end
    if content.tables and char_count < max_chars - 2000:
        parts.append("\n## Tables Found in Protocol\n")
        for i, table in enumerate(content.tables[:10], 1):
            table_entry = f"\n**Table {i}:**\n{table}\n"
            if char_count + len(table_entry) > max_chars:
                break
            parts.append(table_entry)
            char_count += len(table_entry)

    if not parts:
        # Fallback: use raw text
        return content.full_text[:max_chars]

    return "".join(parts)
