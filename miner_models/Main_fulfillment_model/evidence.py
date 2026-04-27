"""
Evidence breakdown builder.

Builds a per-dimension verdict string from pipeline data.
No API calls — purely deterministic from existing data.

Format: "Geography: PASS (reason) | Firmographics: PASS (reason) | ..."
"""

from __future__ import annotations

from typing import Dict, List, Optional


def _verdict(score: float) -> str:
    """Map a 0-1 score to a verdict string."""
    if score >= 0.7:
        return "PASS"
    elif score >= 0.4:
        return "SOFT_PASS"
    elif score >= 0.1:
        return "SOFT_FAIL"
    else:
        return "FAIL"


def _geo_verdict(lead: Dict, icp: Dict) -> str:
    """Geography dimension verdict."""
    target_countries = icp.get("countries", [])
    target_states = icp.get("states", [])
    target_cities = icp.get("cities", [])

    lead_country = (lead.get("country") or "").strip()
    lead_state = (lead.get("state") or "").strip()
    lead_city = (lead.get("city") or "").strip()

    if not target_countries and not target_states and not target_cities:
        return "Geography: PASS (no location filter applied)"

    parts = []
    if target_countries and lead_country:
        if lead_country in target_countries:
            parts.append(f"country '{lead_country}' matches target")
        else:
            parts.append(f"country '{lead_country}' not in target")
    if target_states and lead_state:
        if lead_state in target_states:
            parts.append(f"state '{lead_state}' matches target")
        else:
            parts.append(f"state '{lead_state}' not in target")
    if target_cities and lead_city:
        if lead_city in target_cities:
            parts.append(f"city '{lead_city}' matches target")
        else:
            parts.append(f"city '{lead_city}' not in target cities")

    # Score: all matching = 1.0, partial = 0.5, none = 0.0
    matches = sum(1 for p in parts if "matches" in p)
    total = len(parts) if parts else 1
    score = matches / total
    verdict = _verdict(score)
    reason = "; ".join(parts) if parts else "location data available"

    return f"Geography: {verdict} ({reason})"


def _firmographics_verdict(lead: Dict, icp: Dict) -> str:
    """Firmographics dimension verdict."""
    target_sizes = icp.get("employee_counts", [])
    lead_size = (lead.get("employee_count") or "").strip()

    if not target_sizes:
        return "Firmographics: PASS (no size filter applied)"

    if not lead_size:
        return "Firmographics: SOFT_FAIL (employee count unknown)"

    if lead_size in target_sizes:
        return f"Firmographics: PASS (employee count '{lead_size}' within target range)"

    # Check if any target size partially matches
    return f"Firmographics: SOFT_PASS (employee count '{lead_size}' near target range {', '.join(target_sizes[:3])})"


def _industry_verdict(lead: Dict, icp: Dict) -> str:
    """Industry dimension verdict."""
    target_industries = icp.get("industries", [])
    lead_industry = (lead.get("industry") or "").strip()
    lead_sub = (lead.get("sub_industry") or "").strip()

    if not target_industries:
        return "Industry: PASS (no industry filter applied)"

    if lead_industry in target_industries:
        sub_info = f", sub-industry '{lead_sub}'" if lead_sub else ""
        return f"Industry: PASS (industry '{lead_industry}' matched{sub_info})"

    return f"Industry: FAIL (industry '{lead_industry}' not in target)"


def _persona_verdict(lead: Dict, role_scores: Dict[str, float]) -> str:
    """Persona dimension verdict based on LLM role score."""
    lead_role = (lead.get("role") or "").strip()

    if not lead_role:
        return "Persona: SOFT_FAIL (no role data)"

    if not role_scores:
        return f"Persona: SOFT_PASS (role '{lead_role}' not scored)"

    # Find score for this role
    score = role_scores.get(lead_role, 0.0)
    if score == 0.0:
        # Try case-insensitive match
        for r, s in role_scores.items():
            if r.lower() == lead_role.lower():
                score = s
                break

    verdict = _verdict(score)
    return f"Persona: {verdict} (role '{lead_role}' scored {score:.1f} by LLM)"


def _intent_verdict(intent_data: Dict) -> str:
    """Intent dimension verdict from Perplexity results."""
    intent_score = intent_data.get("intent_score", 0.0)
    signals = intent_data.get("signals", intent_data.get("intent_signals_detail", []))
    signal_count = len(signals) if signals else 0
    matched = sum(1 for s in (signals or []) if s.get("match"))

    # Check cross-source boost
    boost_info = ""
    source_count = intent_data.get("_source_count", 0)
    if source_count >= 2:
        boost_info = f", cross-source boost: {source_count} sources"

    verdict = _verdict(intent_score)
    return f"Intent: {verdict} (intent score {intent_score:.2f} from {signal_count} signals, {matched} matched{boost_info})"


def _data_quality_verdict(verification_data: Dict) -> str:
    """Data quality verdict from email + Stage 4 verification."""
    email_status = verification_data.get("email_status", "unknown")
    stage4_passed = verification_data.get("stage4_passed", None)

    parts = []
    if email_status == "email_ok":
        parts.append("email verified via Truelist")
    elif email_status == "unknown":
        parts.append("email status unknown")
    else:
        parts.append(f"email status: {email_status}")

    if stage4_passed is True:
        parts.append("LinkedIn confirmed via Stage 4")
    elif stage4_passed is False:
        parts.append("Stage 4 verification failed")
    else:
        parts.append("Stage 4 not run")

    # Score: both pass = 1.0, one pass = 0.5, none = 0.0
    score = 0.0
    if email_status == "email_ok":
        score += 0.5
    if stage4_passed:
        score += 0.5

    verdict = _verdict(score)
    return f"Data Quality: {verdict} ({'; '.join(parts)})"


def build_evidence(
    lead: Dict,
    icp_filters: Dict,
    role_scores: Dict[str, float],
    intent_data: Dict,
    verification_data: Dict,
    data_gaps: Optional[List[str]] = None,
) -> str:
    """
    Build per-dimension evidence breakdown string.

    Args:
        lead: Lead dict from DB
        icp_filters: User's ICP (industries, countries, states, etc.)
        role_scores: {role_string: score} from LLM role scoring
        intent_data: Perplexity intent results (intent_score, signals, etc.)
        verification_data: {"email_status": str, "stage4_passed": bool}
        data_gaps: List of data gap descriptions

    Returns:
        Evidence string like "Geography: PASS (...) | Firmographics: PASS (...) | ..."
    """
    parts = [
        _geo_verdict(lead, icp_filters),
        _firmographics_verdict(lead, icp_filters),
        _industry_verdict(lead, icp_filters),
        _persona_verdict(lead, role_scores),
        _intent_verdict(intent_data),
        _data_quality_verdict(verification_data),
    ]

    evidence = " | ".join(parts)

    if data_gaps:
        evidence += "\n---\nData gaps: " + ". ".join(data_gaps) + "."

    return evidence


def compose_intent_details(
    lead_name: str,
    role: str,
    company_name: str,
    employee_count: str,
    sub_industry: str,
    hq_location: str,
    intent_paragraph: str,
) -> str:
    """
    Compose Intent Details column.
    Returns only the Perplexity intent paragraph — person details are in other columns.
    """
    if intent_paragraph and intent_paragraph.strip():
        return intent_paragraph.strip()
    return ""
