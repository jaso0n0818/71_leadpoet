"""
ICP extraction using GPT-5.4 Nano with numbered industry/sub-industry lists.

Two-call approach:
  Call 1: Extract industries (by number) + roles + size + location
  Call 2: Extract sub-industries (by number from matched industries) + expand intent signals

Zero hallucination — LLM picks numbers, we map back to exact taxonomy names.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(__file__))
from industry_taxonomy import INDUSTRY_TAXONOMY

from target_fit_model.config import ICP_PARSER_MODEL
from target_fit_model.openrouter import chat_completion_json

logger = logging.getLogger(__name__)

# Build industry list and sub-industry map from taxonomy
_ALL_INDUSTRIES: List[str] = sorted({ind for v in INDUSTRY_TAXONOMY.values() for ind in v["industries"]})

_INDUSTRY_SUBS: Dict[str, List[str]] = {}
for _sub, _data in INDUSTRY_TAXONOMY.items():
    for _ind in _data["industries"]:
        if _ind not in _INDUSTRY_SUBS:
            _INDUSTRY_SUBS[_ind] = []
        _INDUSTRY_SUBS[_ind].append(_sub)
for _ind in _INDUSTRY_SUBS:
    _INDUSTRY_SUBS[_ind] = sorted(set(_INDUSTRY_SUBS[_ind]))

# Numbered industry list (built once)
_IND_NUMBERED = "\n".join(f"{i+1}. {ind}" for i, ind in enumerate(_ALL_INDUSTRIES))


def parse_icp(
    icp_description: str,
    product_description: Optional[str] = None,
    intent_signals: Optional[str] = None,
) -> Dict:
    """
    Parse ICP from 3 text inputs. Two LLM calls with numbered lists.

    Returns:
        Dict with: industries, sub_industries, roles, countries, states, cities,
        employee_counts, intent_expanded
    """
    icp = icp_description or ""
    product = product_description or ""
    intent = intent_signals or ""

    if not icp and not product:
        return {"error": "At least ICP description or product description is required"}

    # ── CALL 1: Industries + Roles + Size + Location ──
    logger.info("[ICP] Call 1: Extracting industries, roles, size, location...")

    r1 = chat_completion_json(
        prompt=f"""Parse this ICP to find target companies.

IDEAL CUSTOMER PROFILE:
{icp}

PRODUCT & BUYER:
{product}

BUYING SIGNALS:
{intent}

INDUSTRIES (pick by NUMBER):
{_IND_NUMBERED}

COMPANY SIZES: 2-10, 11-50, 51-200, 201-500, 501-1,000, 1,001-5,000, 5,001-10,000, 10,001+

Extract:
1. industry_numbers: What is the PRIMARY industry of the target company? Pick only 1-2 industries that describe what the target company IS. Do NOT pick industries based on job titles, roles, or the product being sold to them. Ignore secondary mentions, contexts, or venues. Return numbers only.
2. roles: Specific searchable job titles mentioned or implied. Not generic ("HR") — use full titles ("VP of Human Resources", "Director of Customer Experience"). Include C-suite who approve purchases.
3. employee_counts: Map any mentioned company sizes to the valid ranges above.
4. countries/states/cities: If locations are mentioned, extract them. If not mentioned, return empty arrays.

Return JSON:
{{"industry_numbers": [5, 19], "roles": ["VP of HR"], "employee_counts": ["51-200"], "countries": [], "states": [], "cities": []}}""",
        model=ICP_PARSER_MODEL,
        system_prompt="Extract structured ICP data. Pick industries by number. Return ONLY valid JSON.",
        temperature=0,
        max_tokens=1000,
    )

    if not r1 or not isinstance(r1, dict):
        logger.error("[ICP] Call 1 failed")
        return {"error": "Failed to parse ICP"}

    # Map industry numbers to names
    industries = []
    for num in (r1.get("industry_numbers") or []):
        if isinstance(num, int) and 1 <= num <= len(_ALL_INDUSTRIES):
            industries.append(_ALL_INDUSTRIES[num - 1])

    roles = r1.get("roles", [])
    employee_counts = r1.get("employee_counts", [])
    raw_countries = r1.get("countries", [])
    states = r1.get("states", [])
    cities = r1.get("cities", [])

    # Normalize country names and expand regions
    _COUNTRY_ALIASES = {
        "uk": "United Kingdom", "u.k.": "United Kingdom", "britain": "United Kingdom",
        "great britain": "United Kingdom", "england": "United Kingdom",
        "usa": "United States", "us": "United States", "u.s.": "United States",
        "america": "United States", "united states of america": "United States",
        "uae": "United Arab Emirates", "emirates": "United Arab Emirates",
        "korea": "South Korea", "holland": "Netherlands", "the netherlands": "Netherlands",
        "deutschland": "Germany", "brasil": "Brazil",
        "czech": "Czech Republic", "czechia": "Czech Republic",
        "russia": "Russia", "russian federation": "Russia",
    }
    _REGION_EXPAND = {
        "europe": ["United Kingdom", "Germany", "France", "Netherlands", "Sweden",
                    "Switzerland", "Spain", "Italy", "Belgium", "Austria", "Denmark",
                    "Norway", "Finland", "Ireland", "Poland", "Portugal", "Czech Republic",
                    "Romania", "Hungary", "Greece", "Luxembourg"],
        "eu": ["Germany", "France", "Netherlands", "Sweden", "Spain", "Italy",
               "Belgium", "Austria", "Denmark", "Finland", "Ireland", "Poland",
               "Portugal", "Czech Republic", "Romania", "Hungary", "Greece", "Luxembourg"],
        "western europe": ["United Kingdom", "Germany", "France", "Netherlands", "Belgium",
                           "Switzerland", "Austria", "Luxembourg", "Ireland"],
        "eastern europe": ["Poland", "Czech Republic", "Romania", "Hungary", "Bulgaria",
                           "Slovakia", "Croatia", "Serbia", "Ukraine", "Lithuania",
                           "Latvia", "Estonia", "Slovenia"],
        "northern europe": ["Sweden", "Norway", "Denmark", "Finland", "Iceland",
                            "Lithuania", "Latvia", "Estonia"],
        "southern europe": ["Spain", "Italy", "Portugal", "Greece", "Croatia", "Serbia"],
        "scandinavia": ["Sweden", "Norway", "Denmark", "Finland", "Iceland"],
        "nordics": ["Sweden", "Norway", "Denmark", "Finland", "Iceland"],
        "dach": ["Germany", "Austria", "Switzerland"],
        "benelux": ["Belgium", "Netherlands", "Luxembourg"],
        "middle east": ["United Arab Emirates", "Saudi Arabia", "Israel", "Qatar",
                        "Kuwait", "Bahrain", "Oman", "Jordan", "Lebanon"],
        "mena": ["United Arab Emirates", "Saudi Arabia", "Israel", "Qatar", "Kuwait",
                 "Egypt", "Morocco", "Tunisia", "Algeria", "Jordan", "Lebanon"],
        "gcc": ["United Arab Emirates", "Saudi Arabia", "Qatar", "Kuwait", "Bahrain", "Oman"],
        "asia": ["China", "Japan", "South Korea", "India", "Singapore", "Indonesia",
                 "Thailand", "Vietnam", "Malaysia", "Philippines", "Taiwan"],
        "southeast asia": ["Singapore", "Indonesia", "Thailand", "Vietnam", "Malaysia",
                           "Philippines", "Myanmar", "Cambodia"],
        "south asia": ["India", "Pakistan", "Bangladesh", "Sri Lanka", "Nepal"],
        "east asia": ["China", "Japan", "South Korea", "Taiwan", "Hong Kong"],
        "apac": ["Australia", "New Zealand", "Japan", "South Korea", "Singapore",
                 "China", "India", "Indonesia", "Thailand", "Vietnam", "Malaysia",
                 "Philippines", "Taiwan", "Hong Kong"],
        "asia pacific": ["Australia", "New Zealand", "Japan", "South Korea", "Singapore",
                         "China", "India", "Indonesia", "Thailand", "Vietnam", "Malaysia"],
        "latam": ["Brazil", "Mexico", "Argentina", "Colombia", "Chile", "Peru",
                  "Ecuador", "Venezuela", "Uruguay", "Costa Rica", "Panama"],
        "latin america": ["Brazil", "Mexico", "Argentina", "Colombia", "Chile", "Peru",
                          "Ecuador", "Venezuela", "Uruguay", "Costa Rica", "Panama"],
        "south america": ["Brazil", "Argentina", "Colombia", "Chile", "Peru",
                          "Ecuador", "Venezuela", "Uruguay", "Paraguay", "Bolivia"],
        "central america": ["Mexico", "Costa Rica", "Panama", "Guatemala",
                            "Honduras", "El Salvador", "Nicaragua", "Belize"],
        "africa": ["South Africa", "Nigeria", "Kenya", "Egypt", "Morocco", "Ghana",
                   "Ethiopia", "Tanzania", "Rwanda", "Senegal"],
        "north america": ["United States", "Canada", "Mexico"],
        "oceania": ["Australia", "New Zealand"],
        "anz": ["Australia", "New Zealand"],
        "uk/europe": ["United Kingdom", "Germany", "France", "Netherlands", "Sweden",
                      "Switzerland", "Spain", "Italy", "Belgium", "Austria", "Denmark",
                      "Norway", "Finland", "Ireland", "Poland"],
        "europe/uk": ["United Kingdom", "Germany", "France", "Netherlands", "Sweden",
                      "Switzerland", "Spain", "Italy", "Belgium", "Austria", "Denmark",
                      "Norway", "Finland", "Ireland", "Poland"],
    }

    countries = []
    for c in raw_countries:
        c_lower = c.lower().strip().rstrip(".")
        if c_lower in _REGION_EXPAND:
            countries.extend(_REGION_EXPAND[c_lower])
        elif c_lower in _COUNTRY_ALIASES:
            countries.append(_COUNTRY_ALIASES[c_lower])
        else:
            # Title case and pass through
            countries.append(c.strip().title())
    # Deduplicate while preserving order
    countries = list(dict.fromkeys(countries))

    logger.info(f"[ICP] Call 1 result: {len(industries)} industries, {len(roles)} roles, {len(countries)} countries")

    if not industries:
        return {
            "error": "Could not identify target industries from description",
            "industries": [], "sub_industries": [], "roles": roles,
            "employee_counts": employee_counts,
            "countries": countries, "states": states, "cities": cities,
            "intent_expanded": intent,
        }

    # ── CALL 2: Sub-industries (numbered) + Intent expansion ──
    logger.info("[ICP] Call 2: Extracting sub-industries and expanding intent...")

    # Build numbered sub-industry list from matched industries
    sub_numbered_lines = []
    sub_map = {}  # number → (sub_name, industry)
    idx = 1
    for ind in industries:
        for sub in _INDUSTRY_SUBS.get(ind, []):
            sub_numbered_lines.append(f"{idx}. {sub} [{ind}]")
            sub_map[idx] = (sub, ind)
            idx += 1

    r2 = chat_completion_json(
        prompt=f"""Pick sub-industries and expand buying signals.

TARGET COMPANIES: {icp}
BUYING SIGNALS: {intent}

SUB-INDUSTRIES (pick by NUMBER — these are the ONLY valid options):
{chr(10).join(sub_numbered_lines)}

Extract:
1. sub_industry_numbers: Pick sub-industries that describe the target company's core business model. Ask: "Would a company in this sub-industry be the buyer described?" If yes, include. If the sub-industry is too specific or too broad for the described buyer, skip it. The target company's industry is what they SELL or PRODUCE, not what they USE, BUY, or are themed around. Numbers only.
2. intent_expanded: Original buying signals + 10-15 additional observable signals that indicate a company needs the product: {product}. Comma-separated.

Return JSON:
{{"sub_industry_numbers": [3, 15], "intent_expanded": "signal1, signal2, ..."}}""",
        model=ICP_PARSER_MODEL,
        system_prompt="Pick sub-industries by number. Expand intent signals. Return ONLY valid JSON.",
        temperature=0,
        max_tokens=1500,
    )

    # Map sub-industry numbers to names
    sub_industries = []
    if r2 and isinstance(r2, dict):
        for num in r2.get("sub_industry_numbers", []):
            if isinstance(num, int) and num in sub_map:
                sub_name, ind = sub_map[num]
                if sub_name in INDUSTRY_TAXONOMY:  # Final validation
                    sub_industries.append(sub_name)

    intent_expanded = ""
    if r2 and isinstance(r2, dict):
        intent_expanded = r2.get("intent_expanded", intent)

    logger.info(f"[ICP] Call 2 result: {len(sub_industries)} sub-industries, intent expanded")

    return {
        "industries": industries,
        "sub_industries": sub_industries,
        "roles": roles,
        "employee_counts": employee_counts,
        "countries": countries,
        "states": states,
        "cities": cities,
        "intent_expanded": intent_expanded,
    }
