"""
Role scoring via LLM.

Scores candidate roles from DB against user's target roles.
Returns {role: score} dict where score is 0.0-1.0.
Same pattern as target fit model's role_matcher.py.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from target_fit_model.config import ICP_PARSER_MODEL
from target_fit_model.openrouter import chat_completion_json

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


def score_roles_via_llm(
    target_roles: List[str],
    candidate_roles: List[str],
    request_description: Optional[str] = None,
    product_description: Optional[str] = None,
    intent_signals: Optional[str] = None,
) -> Dict[str, float]:
    """
    Score candidate DB roles against target roles via LLM.

    Args:
        target_roles: User's desired roles (e.g., ["VP of Sales", "CRO"])
        candidate_roles: Unique roles from DB
        request_description: Target company profile (used if no target_roles)
        product_description: User's product (used if no target_roles)
        intent_signals: Intent signals (used if no target_roles)

    Returns:
        Dict mapping role string to score (0.0-1.0).
        Only includes roles scoring >= 0.5.
    """
    if not candidate_roles:
        return {}

    # If too many roles, chunk into batches
    if len(candidate_roles) > BATCH_SIZE:
        logger.info(f"[RoleScoring] Chunking {len(candidate_roles)} roles into batches of {BATCH_SIZE}")
        all_scored = {}
        for i in range(0, len(candidate_roles), BATCH_SIZE):
            batch = candidate_roles[i:i + BATCH_SIZE]
            batch_scores = _score_single_batch(
                target_roles, batch, request_description, product_description, intent_signals
            )
            all_scored.update(batch_scores)
        logger.info(f"[RoleScoring] Total: {len(all_scored)} scored from {len(candidate_roles)} candidates")
        return all_scored

    return _score_single_batch(
        target_roles, candidate_roles, request_description, product_description, intent_signals
    )


def _score_single_batch(
    target_roles: List[str],
    candidate_roles: List[str],
    request_description: Optional[str] = None,
    product_description: Optional[str] = None,
    intent_signals: Optional[str] = None,
) -> Dict[str, float]:
    """Score a single batch of candidate roles."""
    roles_list = "\n".join(f"- {r}" for r in candidate_roles)

    context_parts = []

    if target_roles:
        target_list = "\n".join(f"- {r}" for r in target_roles)
        context_parts.append(f"The buyer wants to reach people in these target roles:\n{target_list}")

    if product_description:
        context_parts.append(f"Product being sold: {product_description}")
    if request_description:
        context_parts.append(f"Target company profile: {request_description}")
    if intent_signals:
        context_parts.append(f"Intent signals to look for: {intent_signals}")

    context = "\n\n".join(context_parts)

    if target_roles:
        instructions = """From the candidate list below, pick roles that match or are closely related to the target roles above.

Scoring rules:
- 1.0 = exact match (same title as a target role)
- 0.9 = very close variant (e.g. "Senior Head of Design" when target is "Head of Design")
- 0.8 = same function and seniority level (e.g. "Creative Director" when target is "Head of Design")
- 0.7 = related function at leadership level, or CEO/Founder as decision-makers
- 0.6 = tangentially related leadership role
- Below 0.6 = do NOT include

IMPORTANT: Match based on the TARGET ROLES the user specified. The target roles define what functions are relevant — design, marketing, sales, engineering, or anything else. Always respect the user's intent. Do NOT apply any hardcoded function filter."""
    else:
        instructions = """Based on the product and target company description above, identify which candidate roles would be the decision-makers or key stakeholders for purchasing this product.

Scoring rules:
- 1.0 = perfect decision-maker for this product
- 0.9 = very strong match — same function, senior level
- 0.8 = strong match — related function or slightly different seniority
- 0.7 = CEO/President/Owner/Founder (general decision-makers)
- 0.6 = tangentially related leadership role
- Below 0.6 = do NOT include"""

    prompt = f"""{context}

{instructions}

MUST EXCLUDE regardless:
- ANY role with "Representative", "Rep", "Coordinator", "Associate", "Assistant", "Intern" (too junior)
- Roles in completely unrelated functions to what the buyer is looking for

Candidate roles from database:
{roles_list}

Return JSON only: {{"Role Name": score}}
Return {{}} if none match."""

    result = chat_completion_json(
        prompt=prompt,
        model=ICP_PARSER_MODEL,
        system_prompt="You are a B2B lead targeting expert. Score job titles by relevance to the buyer's request. Return ONLY valid JSON.",
        temperature=0,
        max_tokens=8000,
    )

    if not isinstance(result, dict):
        logger.warning("[RoleScoring] LLM returned invalid response")
        return {}

    # Only keep roles that exist in candidate_roles with score >= 0.5
    candidate_set = set(candidate_roles)
    scored = {}
    for r, s in result.items():
        if r in candidate_set:
            try:
                score = float(s)
                if score >= 0.5:
                    scored[r] = score
            except (ValueError, TypeError):
                continue

    logger.info(f"[RoleScoring] {len(scored)} roles scored >= 0.5 from {len(candidate_roles)} candidates")
    return scored


def get_ranked_roles(scored_roles: Dict[str, float], target_roles: Optional[List[str]] = None) -> List[str]:
    """Return role strings sorted by score descending, with exact target role matches first."""
    target_lower = {t.lower() for t in target_roles} if target_roles else set()

    def sort_key(item):
        role, score = item
        is_exact = 1 if role.lower() in target_lower else 0
        return (-is_exact, -score)

    return [r for r, _ in sorted(scored_roles.items(), key=sort_key)]
