"""
Fulfillment scoring pipeline: three-tier gate-then-score architecture.

Tier 1: ICP Fit Gate ($0 — free exact-match checks)
Tier 2: Data Accuracy Gate ($low — external API calls, no LLM)
Tier 3: Intent Scoring ($moderate — LLM calls, peak-weighted aggregation)
"""

import logging
import re
from typing import List, Optional, Set, Tuple

from gateway.fulfillment.config import (
    get_fulfillment_api_key,
    FULFILLMENT_MIN_INTENT_SCORE,
    FULFILLMENT_INTENT_QUALITY_FLOOR,
    FULFILLMENT_INTENT_BREADTH_WEIGHT,
)
from gateway.fulfillment.models import (
    FulfillmentLead,
    FulfillmentICP,
    FulfillmentScoreResult,
    VALID_ROLE_TYPES,
)
from gateway.qualification.models import LeadOutput, ICPPrompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Peak-weighted intent aggregation
# ---------------------------------------------------------------------------

def aggregate_intent_scores(signal_scores: List[float]) -> float:
    """
    Peak-weighted aggregation: best signal dominates, quality signals
    add diminishing breadth bonus, noise is ignored.
    """
    if not signal_scores:
        return 0.0

    sorted_desc = sorted(signal_scores, reverse=True)
    best = sorted_desc[0]

    bonus = 0.0
    for i, score in enumerate(sorted_desc[1:], start=1):
        if score < FULFILLMENT_INTENT_QUALITY_FLOOR:
            break
        bonus += score * FULFILLMENT_INTENT_BREADTH_WEIGHT * (1 / i)

    return min(best + bonus, 60.0)


# ---------------------------------------------------------------------------
# Employee count range helpers
# ---------------------------------------------------------------------------

def _parse_employee_range(val: str) -> Tuple[int, int]:
    """Parse employee count string into (min, max). Returns (0, 0) on failure."""
    if not val:
        return (0, 0)
    val = val.strip()
    if re.match(r"^\d+$", val):
        n = int(val)
        return (n, n)
    m = re.match(r"^(\d+)-(\d+)$", val)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.match(r"^(\d+)\+$", val)
    if m:
        return (int(m.group(1)), 10_000_000)
    return (0, 0)


def _ranges_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


# ---------------------------------------------------------------------------
# Country normalization (reuses existing logic)
# ---------------------------------------------------------------------------

def _normalize_country(c: str) -> str:
    """Simple country alias normalization."""
    aliases = {
        "us": "united states", "usa": "united states", "u.s.": "united states",
        "u.s.a.": "united states", "uk": "united kingdom",
        "gb": "united kingdom", "great britain": "united kingdom",
    }
    c = c.strip().lower()
    return aliases.get(c, c)


# ---------------------------------------------------------------------------
# Tier 1: ICP Fit Gate (free, deterministic)
# ---------------------------------------------------------------------------

def _tier1_check(
    lead: FulfillmentLead,
    lead_output: LeadOutput,
    icp: FulfillmentICP,
    seen_companies: Set[str],
) -> Optional[str]:
    """
    Return failure_reason string if the lead fails any ICP check, else None.
    """
    if icp.industry and lead.industry != icp.industry:
        return "industry_mismatch"

    if icp.sub_industry and lead.sub_industry != icp.sub_industry:
        return "sub_industry_mismatch"

    if icp.target_role_types and lead.role_type not in icp.target_role_types:
        return "role_type_mismatch"

    if icp.target_roles and lead.role not in icp.target_roles:
        return "role_mismatch"

    if icp.target_seniority:
        try:
            lead_sen = lead_output.seniority.value if hasattr(lead_output.seniority, "value") else str(lead_output.seniority)
            if lead_sen.lower() != icp.target_seniority.lower():
                return "seniority_mismatch"
        except Exception:
            return "seniority_mismatch"

    if icp.country and lead.company_hq_country:
        if _normalize_country(lead.company_hq_country) != _normalize_country(icp.country):
            return "country_mismatch"

    if icp.employee_count and lead.employee_count:
        icp_range = _parse_employee_range(icp.employee_count)
        lead_range = _parse_employee_range(lead.employee_count)
        if icp_range != (0, 0) and lead_range != (0, 0):
            if not _ranges_overlap(icp_range, lead_range):
                return "employee_count_mismatch"

    if icp.company_stage and lead_output.role:
        pass

    biz_lower = lead_output.business.strip().lower()
    if not biz_lower:
        return "data_quality"
    if biz_lower in seen_companies:
        return "duplicate_company"
    seen_companies.add(biz_lower)

    return None


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------

async def score_fulfillment_lead(
    lead: FulfillmentLead,
    icp: FulfillmentICP,
    seen_companies: Set[str],
    email_results: Optional[dict] = None,
) -> FulfillmentScoreResult:
    """Score a single fulfillment lead through the three-tier pipeline."""
    lead_output = lead.to_lead_output()
    icp_prompt = icp.to_icp_prompt()

    # --- Tier 1 ---
    t1_failure = _tier1_check(lead, lead_output, icp, seen_companies)
    if t1_failure:
        return FulfillmentScoreResult(
            tier1_passed=False, tier2_passed=False,
            failure_reason=t1_failure,
        )

    # --- Tier 2 (data accuracy) ---
    try:
        t2_failure = await _run_tier2(lead, email_results)
        if t2_failure:
            return FulfillmentScoreResult(
                tier1_passed=True, tier2_passed=False,
                failure_reason=t2_failure,
            )
    except Exception as e:
        logger.warning(f"Tier 2 error (soft pass): {e}")

    # --- Tier 3 (intent scoring) ---
    api_key = get_fulfillment_api_key()
    from qualification.scoring.lead_scorer import (
        _score_single_intent_signal,
        _apply_signal_time_decay,
        _extract_domain,
    )

    icp_criteria = None
    seen_domains: Set[str] = set()
    signal_results = []

    for signal in lead.intent_signals:
        domain = _extract_domain(signal.url)
        if domain in seen_domains:
            signal_results.append({"after_decay": 0.0, "decay_mult": 1.0, "confidence": 0})
            continue
        seen_domains.add(domain)

        try:
            score, confidence, date_status, content_found_date = await _score_single_intent_signal(
                signal, icp_prompt, icp_criteria,
                lead_output.business, lead_output.company_website,
                api_key=api_key,
            )
        except Exception as e:
            logger.warning(f"Signal scoring error: {e}")
            score, confidence, date_status, content_found_date = 0.0, 0, "fabricated", None

        source_str = signal.source.value if hasattr(signal.source, "value") else str(signal.source)
        after_decay, decay_mult = _apply_signal_time_decay(
            score, signal.date, date_status, source_str, content_found_date
        )

        signal_results.append({
            "after_decay": after_decay,
            "decay_mult": decay_mult,
            "confidence": confidence,
        })

    after_decay_scores = [r["after_decay"] for r in signal_results]
    intent_signal_final = aggregate_intent_scores(after_decay_scores)
    intent_signal_final = min(intent_signal_final, 60.0)

    all_fabricated = bool(signal_results) and all(r["confidence"] == 0 for r in signal_results)

    if intent_signal_final < FULFILLMENT_MIN_INTENT_SCORE:
        return FulfillmentScoreResult(
            tier1_passed=True, tier2_passed=True,
            intent_signal_raw=max(after_decay_scores) if after_decay_scores else 0.0,
            intent_signal_final=intent_signal_final,
            intent_decay_multiplier=_avg([r["decay_mult"] for r in signal_results]),
            final_score=0.0,
            all_fabricated=all_fabricated,
            failure_reason="insufficient_intent",
        )

    return FulfillmentScoreResult(
        tier1_passed=True,
        tier2_passed=True,
        intent_signal_raw=max(after_decay_scores) if after_decay_scores else 0.0,
        intent_signal_final=intent_signal_final,
        intent_decay_multiplier=_avg([r["decay_mult"] for r in signal_results]),
        final_score=intent_signal_final,
        all_fabricated=all_fabricated,
    )


async def score_fulfillment_batch(
    leads: List[FulfillmentLead],
    icp: FulfillmentICP,
) -> List[FulfillmentScoreResult]:
    """Score a batch of fulfillment leads with cross-lead dedup and batch email verification."""
    seen_companies: Set[str] = set()
    results: List[FulfillmentScoreResult] = []

    for lead in leads:
        result = await score_fulfillment_lead(lead, icp, seen_companies)
        results.append(result)

    # Structural similarity detection on leads that passed all tiers
    try:
        from qualification.scoring.lead_scorer import detect_structural_similarity
        passing_outputs = []
        passing_indices = []
        for i, (lead, result) in enumerate(zip(leads, results)):
            if result.final_score > 0:
                passing_outputs.append(lead.to_lead_output())
                passing_indices.append(i)

        if len(passing_outputs) >= 2:
            flagged = detect_structural_similarity(passing_outputs)
            for local_idx in flagged:
                global_idx = passing_indices[local_idx]
                results[global_idx] = FulfillmentScoreResult(
                    **{
                        **results[global_idx].model_dump(),
                        "final_score": 0.0,
                        "failure_reason": "structural_similarity_detected",
                    }
                )
    except Exception as e:
        logger.warning(f"Structural similarity detection error: {e}")

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _run_tier2(lead: FulfillmentLead, email_results: Optional[dict] = None) -> Optional[str]:
    """
    Run Tier 2 data accuracy checks. Returns failure reason or None.
    Imports are deferred to avoid hard dependency on validator_models at import time.
    """
    try:
        from validator_models.automated_checks import run_stage0_2_checks
    except ImportError:
        logger.warning("validator_models not available — skipping Tier 2")
        return None

    lead_dict = lead.model_dump()

    try:
        stage0_2 = run_stage0_2_checks(lead_dict)
        if isinstance(stage0_2, dict) and stage0_2.get("hard_failure"):
            return stage0_2.get("reason", "stage0_2_failure")
    except Exception as e:
        logger.warning(f"Stage 0-2 error (soft pass): {e}")

    return None


def _avg(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0
