"""
Fit scoring engine for Target-Fit Lead Model.

Scores each candidate lead against a buyer's ICP across multiple dimensions.
Each dimension returns 0.0-1.0, weighted and combined into a final fit_score.

No database access — pure computation on in-memory lead dicts.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from target_fit_model.config import (
    WEIGHT_INDUSTRY,
    WEIGHT_SUB_INDUSTRY,
    WEIGHT_ROLE,
    WEIGHT_LOCATION,
    WEIGHT_COMPANY_SIZE,
    WEIGHT_QUALITY,
    EMPLOYEE_RANGE_ORDER,
    EMPLOYEE_RANGE_ALIASES,
    SENIORITY_RANK,
    ROLE_KEYWORD_EXPANSIONS,
    ROLE_STOPWORDS,
    SENIORITY_KEYWORDS,
    COUNTRY_ALIASES,
    MAX_REP_SCORE,
)


def compute_fit_score(lead: Dict, icp: Dict) -> Tuple[float, Dict]:
    """
    Score a lead against the buyer's ICP.

    Args:
        lead: Lead dict with company/contact fields
        icp: Buyer criteria dict with keys:
             industry, sub_industry, role, country, state, city, employee_count

    Returns:
        (fit_score, breakdown) where fit_score is 0.0-1.0 and breakdown
        shows the per-dimension scores.
    """
    ind = _score_industry(lead, icp)
    sub = _score_sub_industry(lead, icp)
    role = _score_role(lead, icp)
    loc = _score_location(lead, icp)
    size = _score_company_size(lead, icp)
    qual = _score_quality(lead)

    fit_score = (
        ind * WEIGHT_INDUSTRY
        + sub * WEIGHT_SUB_INDUSTRY
        + role * WEIGHT_ROLE
        + loc * WEIGHT_LOCATION
        + size * WEIGHT_COMPANY_SIZE
        + qual * WEIGHT_QUALITY
    )

    breakdown = {
        "industry_match": round(ind, 3),
        "sub_industry_match": round(sub, 3),
        "role_match": round(role, 3),
        "location_match": round(loc, 3),
        "size_match": round(size, 3),
        "quality_bonus": round(qual, 3),
    }

    return round(fit_score, 4), breakdown


# ═══════════════════════════════════════════════════════════════════════════
# Dimension Scorers
# ═══════════════════════════════════════════════════════════════════════════

def _score_industry(lead: Dict, icp: Dict) -> float:
    """Industry is a hard filter, so this is always 1.0 for queried leads."""
    lead_ind = (lead.get("industry") or "").strip().lower()
    targets = [t.strip().lower() for t in icp.get("industries", []) if t.strip()]
    if not targets:
        return 1.0
    return 1.0 if lead_ind in targets else 0.0


def _score_sub_industry(lead: Dict, icp: Dict) -> float:
    targets = [t.strip().lower() for t in icp.get("sub_industries", []) if t.strip()]
    if not targets:
        return 0.5
    lead_sub = (lead.get("sub_industry") or "").strip().lower()
    if lead_sub in targets:
        return 1.0
    best = max((_word_overlap(lead_sub, t) for t in targets), default=0.0)
    if best > 0.4:
        return 0.7
    return 0.1


from target_fit_model.query import TITLE_EQUIVALENTS, FUNCTION_SYNONYMS


def _get_all_match_terms(target: str) -> list:
    """Get all terms that should match for a given role input.

    Three types:
    1. Title equivalents: 'CEO' → ['ceo', 'chief executive officer']
    2. Title + dept: 'VP of Sales' → ['vp of sales', 'vice president of sales']
    3. Function synonyms: 'marketing' → ['marketing', 'brand', 'demand generation', ...]
    """
    key = target.strip().lower()
    all_terms = []

    for canonical, equivalents in TITLE_EQUIVALENTS.items():
        for eq in equivalents:
            if key == eq:
                all_terms.extend(equivalents)
                return all_terms
            if key.startswith(eq + " "):
                suffix = key[len(eq):]
                all_terms.extend([e + suffix for e in equivalents])
                return all_terms

    for canonical, synonyms in FUNCTION_SYNONYMS.items():
        for syn in synonyms:
            if key == syn or key == canonical:
                all_terms.extend(synonyms)
                return all_terms
            if key.startswith(syn + " ") or (len(syn) > 3 and syn in key.split()):
                all_terms.extend(synonyms)
                return all_terms

    return [key]


def _score_role(lead: Dict, icp: Dict) -> float:
    """
    Role matching: does the lead's role contain any of the target terms
    (including equivalents and synonyms)?

    'CEO' matches CEO, Chief Executive Officer, CEO & Founder, President & CEO.
    'PR' matches Public Relations Manager, PR Director, Communications Director.
    'VP of Sales' matches VP of Sales, Vice President of Sales.
    """
    targets = [t.strip() for t in icp.get("roles", []) if t.strip()]
    if not targets:
        return 0.5

    lead_role = (lead.get("role") or "").strip()
    if not lead_role:
        return 0.0

    lead_lower = lead_role.lower()

    for target in targets:
        match_terms = _get_all_match_terms(target)
        for term in match_terms:
            if term in lead_lower:
                return 1.0

    return 0.0


def _score_location(lead: Dict, icp: Dict) -> float:
    target_countries = [_normalize_country(c) for c in icp.get("countries", []) if c.strip()]
    target_states = [s.strip().lower() for s in icp.get("states", []) if s.strip()]
    target_cities = [c.strip().lower() for c in icp.get("cities", []) if c.strip()]

    if not target_countries and not target_states and not target_cities:
        return 0.5

    lead_country = _normalize_country(lead.get("country") or "")
    lead_state = (lead.get("state") or "").strip().lower()
    lead_city = (lead.get("city") or "").strip().lower()

    score = 0.0
    checks = 0

    if target_countries:
        checks += 1
        if lead_country in target_countries:
            score += 1.0

    if target_states:
        checks += 1
        if lead_state in target_states:
            score += 1.0

    if target_cities:
        checks += 1
        if lead_city in target_cities:
            score += 1.0
        elif any(tc in lead_city or lead_city in tc for tc in target_cities):
            score += 0.7

    return score / checks if checks > 0 else 0.5


def _score_company_size(lead: Dict, icp: Dict) -> float:
    targets = [t.strip() for t in icp.get("employee_counts", []) if t.strip()]
    if not targets:
        return 0.5

    lead_size = (lead.get("employee_count") or "").strip()
    if not lead_size:
        return 0.2

    lead_idx = _employee_range_index(lead_size)
    if lead_idx is None:
        return 0.3

    best = 0.0
    for target_size in targets:
        icp_idx = _employee_range_index(target_size)
        if icp_idx is None:
            continue
        distance = abs(icp_idx - lead_idx)
        if distance == 0:
            s = 1.0
        elif distance == 1:
            s = 0.7
        elif distance == 2:
            s = 0.4
        else:
            s = 0.1
        if s > best:
            best = s
    return best if best > 0 else 0.3


def _score_quality(lead: Dict) -> float:
    rep = lead.get("rep_score")
    if rep is None:
        return 0.0
    try:
        rep = float(rep)
    except (ValueError, TypeError):
        return 0.0
    return min(max(rep / MAX_REP_SCORE, 0.0), 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Role Matching Helpers (adapted from champion model)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_role_keywords(role: str) -> set:
    words = set(role.lower().split())
    keywords = words - ROLE_STOPWORDS

    role_lower = role.lower()
    for abbrev, expansions in ROLE_KEYWORD_EXPANSIONS.items():
        if abbrev in role_lower.split():
            keywords |= expansions

    return keywords


def _role_keyword_score(lead_role: str, icp_role: str) -> float:
    lead_kw = _extract_role_keywords(lead_role)
    icp_kw = _extract_role_keywords(icp_role)

    if not icp_kw or not lead_kw:
        return 0.3

    overlap = lead_kw & icp_kw
    union = lead_kw | icp_kw
    jaccard = len(overlap) / len(union) if union else 0.0

    if overlap:
        return min(0.3 + jaccard * 0.7, 1.0)
    return 0.0


def _detect_seniority(role: str) -> str:
    role_lower = role.lower()
    for level, keywords in SENIORITY_KEYWORDS.items():
        for kw in keywords:
            if kw in role_lower:
                return level
    return "Individual Contributor"


def _role_seniority_score(lead_role: str, icp_role: str) -> float:
    lead_level = _detect_seniority(lead_role)
    icp_level = _detect_seniority(icp_role)

    lead_rank = SENIORITY_RANK.get(lead_level, 1)
    icp_rank = SENIORITY_RANK.get(icp_level, 1)

    distance = abs(lead_rank - icp_rank)
    if distance == 0:
        return 1.0
    elif distance == 1:
        return 0.7
    elif distance == 2:
        return 0.4
    else:
        return 0.1


# ═══════════════════════════════════════════════════════════════════════════
# General Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_country(name: str) -> str:
    stripped = name.strip().lower()
    return COUNTRY_ALIASES.get(stripped, stripped)


def _word_overlap(a: str, b: str) -> float:
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _employee_range_index(raw: str) -> Optional[int]:
    normalized = raw.strip().lower()
    normalized = EMPLOYEE_RANGE_ALIASES.get(normalized)
    if normalized is None:
        normalized = EMPLOYEE_RANGE_ALIASES.get(raw.strip())
    if normalized is None:
        cleaned = re.sub(r'[,\s]', '', raw.lower()).replace('employees', '')
        for alias, canonical in EMPLOYEE_RANGE_ALIASES.items():
            alias_clean = re.sub(r'[,\s]', '', alias.lower()).replace('employees', '')
            if alias_clean == cleaned:
                normalized = canonical
                break
    if normalized is None:
        return None
    try:
        return EMPLOYEE_RANGE_ORDER.index(normalized)
    except ValueError:
        return None
