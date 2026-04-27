"""
Target-role auto-expansion for fulfillment requests.

Called once per request at creation time to expand a small seed list of
``target_roles`` (e.g. ``["VP of Sales"]``) into a broader list of common
variant spellings and near-synonyms (``["VP of Sales", "VP Sales",
"Vice President of Sales", "Sales VP", "Head of Sales", ...]``).

Downstream matching is a plain case-insensitive string equality check
between the miner-submitted lead role and the ICP's ``target_roles``
list (see ``gateway/fulfillment/scoring.py::_tier1_check``).  A tiny
seed list therefore rejects many legitimate leads whose LinkedIn titles
use a slightly different wording — "Director of Sales" for "VP of
Sales" at a small company, or "VP, Sales" (comma vs "of") at an
enterprise.

Expanding once at request creation puts the cost on the request side
(one LLM call per request, ~20/day) instead of the per-lead scoring
side (one call per lead, potentially thousands per day).  The expanded
list is then locked into ``icp_details`` before hashing so every miner
and validator sees the same canonical list for the life of the
request.

Guarantees:
  * Never DROPS a caller-supplied role — the original list is always
    included in the result.
  * Never EXPANDS scope — the prompt constrains the model to same-
    seniority / same-function variants (so "VP of Sales" does not pull
    in "Director of Sales" or "Head of Revenue").
  * On any LLM / parse failure, returns the original list unchanged.
    Caller does not need to try/except — this is best-effort.
  * Caps the output at :data:`MAX_EXPANDED_ROLES` so a prompt-injection
    or runaway LLM output can't blow up icp_details.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import List

import httpx

logger = logging.getLogger(__name__)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# gpt-4o-mini is a small / cheap model and still very good at synonym
# generation.  Using a bigger model (e.g. sonar-pro) here would just
# burn latency + $ for no measurable quality gain on this task.
MODEL = "openai/gpt-4o-mini"
TIMEOUT_SECONDS = 20
MAX_EXPANDED_ROLES = 30  # hard cap: prevents abuse / runaway outputs


_PROMPT_TEMPLATE = """You expand job-title seed lists into a canonical set of variant titles that an ATS or LinkedIn member might use to describe the same role at the same seniority.

Input seed titles: {roles}
{seniority_hint}

Rules:
- Output ONLY a JSON array of strings.  No prose, no code fences.
- INCLUDE every seed title verbatim.
- Add common variant spellings, word orders, and near-synonyms that are the SAME seniority and SAME function.  Examples of valid expansions of "VP of Sales": "VP Sales", "Vice President of Sales", "VP, Sales", "Sales VP", "Senior VP of Sales", "SVP Sales", "SVP, Sales".
- Do NOT include roles ONE LEVEL UP (CRO, Chief Revenue Officer, Chief Sales Officer).
- Do NOT include roles ONE LEVEL DOWN (Director of Sales, Sales Manager, Sales Lead).
- Do NOT include different functions (e.g. VP of Engineering when the seed is VP of Sales).
- Target 10-15 entries per seed title.  Total output must not exceed {max_total} entries.
- Titles should be typical casing; do not ALL-CAPS anything that isn't an abbreviation.

Respond with the JSON array only."""


async def expand_target_roles(
    roles: List[str],
    target_seniority: str = "",
    api_key: str = "",
) -> List[str]:
    """Return an expanded role list.  On any failure returns the input.

    Args:
        roles: caller's seed list of target role titles.  Preserved
            verbatim in the output.
        target_seniority: optional seniority hint (e.g. ``"VP"``,
            ``"Director"``).  Passed to the model as a hint so it
            doesn't expand across seniority lines.  Empty string is
            fine — the per-role prompt rules still hold.
        api_key: OpenRouter API key.  If empty, falls back to the
            ``FULFILLMENT_OPENROUTER_API_KEY`` env var; if that is
            also empty, returns the input unchanged.
    """
    if not roles:
        return roles

    # Normalize + dedupe the input seeds first (client might have typed
    # duplicates), keeping first-seen ordering for the verbatim output.
    seeds: List[str] = []
    seen_keys = set()
    for r in roles:
        if not isinstance(r, str):
            continue
        clean = r.strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        seeds.append(clean)

    if not seeds:
        return roles

    key = api_key or os.getenv("FULFILLMENT_OPENROUTER_API_KEY", "")
    if not key:
        logger.warning(
            "expand_target_roles: FULFILLMENT_OPENROUTER_API_KEY not set, "
            "returning input unchanged"
        )
        return list(seeds)

    seniority_hint = (
        f"Target seniority: {target_seniority}" if target_seniority else ""
    )
    prompt = _PROMPT_TEMPLATE.format(
        roles=json.dumps(seeds),
        seniority_hint=seniority_hint,
        max_total=MAX_EXPANDED_ROLES,
    )

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
            body = resp.json()
    except Exception as e:
        logger.warning(
            f"expand_target_roles: LLM call failed, returning seeds unchanged: "
            f"{type(e).__name__}: {e}"
        )
        return list(seeds)

    try:
        text = body["choices"][0]["message"]["content"] or ""
    except Exception as e:
        logger.warning(f"expand_target_roles: unexpected response shape: {e}")
        return list(seeds)

    variants = _parse_variants(text)
    if not variants:
        logger.warning(
            "expand_target_roles: LLM returned no parseable variants, "
            "returning seeds unchanged"
        )
        return list(seeds)

    # Merge: seeds first, then LLM variants, deduped case-insensitively.
    merged: List[str] = []
    seen = set()
    for item in list(seeds) + list(variants):
        if not isinstance(item, str):
            continue
        clean = item.strip()
        if not clean:
            continue
        k = clean.lower()
        if k in seen:
            continue
        seen.add(k)
        merged.append(clean)
        if len(merged) >= MAX_EXPANDED_ROLES:
            break

    return merged


def _parse_variants(text: str) -> List[str]:
    """Extract a list of strings from the LLM response.

    Tries strict JSON first, then falls back to finding the first
    top-level ``[...]`` block and parsing that.  Silently returns
    ``[]`` on any failure — caller handles the empty-list case.
    """
    text = text.strip()
    # Strip accidental code fences.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    # Try full parse.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if x is not None]
    except Exception:
        pass

    # Fallback: pull the first [...] block.
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(0))
        if isinstance(parsed, list):
            return [str(x) for x in parsed if x is not None]
    except Exception:
        pass

    return []
