"""
Funding intent detection and company funding verification.

When user's intent signals mention funding, this module:
1. Classifies the funding criteria (type, timeframe)
2. Checks each company against those criteria via Perplexity
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from target_fit_model.config import ICP_PARSER_MODEL, PERPLEXITY_MODEL, PERPLEXITY_TIMEOUT
from target_fit_model.openrouter import chat_completion, chat_completion_json

logger = logging.getLogger(__name__)

FUNDING_KEYWORDS = [
    "funded", "funding", "raised", "series a", "series b", "series c", "series d",
    "seed", "investment", "round", "venture capital", "vc backed", "vc-backed",
    "recently funded", "just raised", "capital raise", "fundraise",
]


def detect_funding_intent(intent_signals: str) -> bool:
    """Check if intent signals mention funding."""
    if not intent_signals:
        return False
    lower = intent_signals.lower()
    return any(kw in lower for kw in FUNDING_KEYWORDS)


def classify_funding_criteria(intent_signals: str) -> Optional[Dict]:
    """
    Use GPT-5.4 Nano to extract specific funding criteria from intent signals.
    Returns: {"funding_type": "Series A-C", "timeframe_days": 90, "description": "..."}
    """
    result = chat_completion_json(
        prompt=f"""Extract the funding criteria from this intent signal:

"{intent_signals}"

What kind of funding is the user looking for? Extract:
1. funding_type: What rounds? (Seed, Series A, Series B, Series C, any, etc.)
2. timeframe_days: How recent? Convert to days. "last 90 days" = 90, "last 6 months" = 180, "last year" = 365. If not specified, default to 180.
3. description: A natural language description of the funding criteria for a research query.

Return JSON:
{{"funding_type": "Series A-C or Seed", "timeframe_days": 90, "description": "raised Series A, B, or C funding in the last 90 days"}}""",
        model=ICP_PARSER_MODEL,
        system_prompt="Extract funding criteria. Return ONLY valid JSON.",
        temperature=0,
        max_tokens=300,
    )

    if result and isinstance(result, dict):
        logger.info(f"[Funding] Criteria: {result}")
        return result

    # Default fallback
    return {"funding_type": "any", "timeframe_days": 180, "description": "raised funding in the last 6 months"}


def check_company_funding(
    company_name: str,
    website: str,
    funding_criteria: Dict,
) -> Tuple[bool, str]:
    """
    Check if a single company matches funding criteria via Perplexity.
    Returns: (passed: bool, evidence: str)
    """
    description = funding_criteria.get("description", "raised funding recently")
    timeframe = funding_criteria.get("timeframe_days", 180)

    result = chat_completion_json(
        prompt=f"""Check if this company recently received funding.

Company: "{company_name}"
Website: {website}

Criteria: {description}

Search for: funding announcements, press releases, Crunchbase, PitchBook, TechCrunch, news articles.

Return JSON:
{{
  "funded": true or false,
  "round_type": "Series A" or null,
  "amount": "$10M" or null,
  "date": "January 2026" or null,
  "evidence": "brief description with source URL, or null if not found"
}}

Only return funded=true if you find CONFIRMED evidence from the last {timeframe} days. Do not guess.""",
        model=PERPLEXITY_MODEL,
        system_prompt="Check company funding status. Return ONLY valid JSON.",
        temperature=0,
        max_tokens=500,
        timeout=PERPLEXITY_TIMEOUT,
    )

    if result and isinstance(result, dict):
        funded = result.get("funded", False)
        evidence = result.get("evidence") or ""
        date_str = result.get("date") or ""
        if funded:
            parts = []
            if result.get("round_type"):
                parts.append(result["round_type"])
            if result.get("amount"):
                parts.append(result["amount"])
            if date_str:
                parts.append(date_str)
            if evidence:
                parts.append(evidence)
            full_evidence = " — ".join(parts) if parts else "Funding confirmed"
            return True, full_evidence, date_str
        return False, "", ""

    return False, "", ""


def _parse_date_to_sortkey(date_str: str) -> int:
    """Convert date string like 'February 2026' to a sortable int (most recent = highest)."""
    import re
    from datetime import datetime

    if not date_str:
        return 0

    # Try common formats
    for fmt in ["%B %Y", "%b %Y", "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%Y"]:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return int(dt.strftime("%Y%m%d"))
        except ValueError:
            continue

    # Try extracting year and month with regex
    match = re.search(r"(20\d{2})", date_str)
    if match:
        year = int(match.group(1))
        month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                     "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
        for m, n in month_map.items():
            if m in date_str.lower():
                return year * 100 + n
        return year * 100

    return 0


def batch_check_funding(
    companies: List[Dict],
    funding_criteria: Dict,
    target_count: int,
) -> List[Dict]:
    """
    Check companies one by one until target_count funded companies found.
    Returns list sorted by funding recency (most recent first).
    """
    funded_companies = []
    checked = 0

    for company in companies:
        if len(funded_companies) >= target_count:
            break

        company_name = company.get("company_name", "")
        website = company.get("website", "")
        checked += 1

        logger.info(f"[Funding] Checking {checked}/{len(companies)}: {company_name} "
                     f"({len(funded_companies)}/{target_count} funded)")

        passed, evidence, date_str = check_company_funding(company_name, website, funding_criteria)

        if passed:
            company["_funding_passed"] = True
            company["_funding_evidence"] = evidence
            company["_funding_date"] = date_str
            company["_funding_sort"] = _parse_date_to_sortkey(date_str)
            funded_companies.append(company)
            logger.info(f"[Funding] ✓ {company_name}: {evidence}")
        else:
            logger.info(f"[Funding] ✗ {company_name}: no funding found")

    # Sort by most recently funded first
    funded_companies.sort(key=lambda c: c.get("_funding_sort", 0), reverse=True)

    logger.info(f"[Funding] Result: {len(funded_companies)}/{target_count} funded from {checked} checked")
    return funded_companies
