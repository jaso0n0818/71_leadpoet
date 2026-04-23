"""
Intent Details passage generator (LLM).

Runs on the gateway once per winning fulfillment lead, after consensus.
Synthesizes the miner's verified intent signals into a single
client-ready paragraph for the "Intent Details" UI column.

Model: ``openai/gpt-4o-mini`` via OpenRouter.
Key:   ``FULFILLMENT_OPENROUTER_API_KEY`` (already wired into the gateway env).

Why not Perplexity sonar-pro (previous model)?  sonar-pro is a
web-search-augmented model — when its search returns no hits for the
company name, it *refuses* and emits strings like "I need to clarify..."
or "the search results don't contain information about...".  That's the
correct bias for an open-web Q&A product but wrong for us: we already
have the evidence (miner-submitted intent signals with URLs, snippets,
and dates) and just want faithful synthesis.  gpt-4o-mini is non-search,
deterministic-leaning, cheaper, faster, and much better at staying
strictly inside the provided context.

This module is self-contained: no dependency on miner-side helpers
(``target_fit_model`` / ``openrouter.py``) which aren't deployed on the
gateway.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-4o-mini"
LLM_TIMEOUT_SECONDS = 60


# Prompt text closely mirrors the exact rule list the client uses when
# hand-crafting Intent Details in Perplexity.  If you adjust a rule, this
# should be a product decision synced with whatever's in the client-facing
# UI spec, not a silent code change.
_PROMPT_TEMPLATE = """Fill in the Intent Details paragraph for one specific company using the provided ICP and the identified intent signals below.

Rules (strictly enforced):
- Only produce the Intent Details paragraph. No preamble, no apology, no disclaimer, no mention of "search results", "available information", or your own limitations.
- Synthesize ONLY from the intent signals provided below. Do not invent facts, do not speculate beyond what the signals support, and do not reference any source outside the inputs.
- Intent Details must be expanded with rich buying-signal context explaining why the observed activity indicates relevance for the ICP.
- Write as a single natural paragraph. No subtitles, bullets, labels, or markdown.
- Do not reference "this list", "these candidates", or the source file.
- Do not add links, citations, or references.
- Output must be client-ready for direct use in the UI.
- Use natural paragraph prose. No em dashes. Avoid over-using hyphens.
- Do not restate the client name or ICP facts (country, employee count, industry, etc.).
- Ensure every claim is grounded in the provided evidence. If a specific claim cannot be supported by the provided signals, omit it — write a shorter paragraph rather than fabricating detail.

Inputs:

ICP:
{icp_block}

Intent Signals (the only evidence you may cite):
{signals_block}

Output:

A single polished paragraph for the Intent Details column that clearly explains the relevance and strength of this company's buying intent, grounded in the provided signals."""


# Patterns that indicate the LLM emitted meta-commentary / refusal instead
# of a real passage.  If the first ~200 characters match any of these, we
# discard the output and return empty — the caller (lifecycle) treats that
# as "skip this lead, don't persist intent_details".
_REFUSAL_RE = re.compile(
    r"\b(i need to clarify|i appreciate (the )?(detailed )?request|"
    r"i apologi[sz]e|i(?:'m| am) (unable|sorry|not able)|"
    r"unfortunately (i|we) (cannot|can't|don't|do not)|"
    r"i (cannot|can't|don't|do not) (generate|produce|find|have)|"
    r"the (provided )?(search results|available information|given "
    r"(data|information|context|signals|search results)) (do not|does not|don't|doesn't) "
    r"contain|no (verifiable |factual |specific |explicit )?"
    r"(information|data|evidence|details) (about|on|regarding|for) "
    r"[a-z0-9][\w\s&.,'()-]*? (is|was|has been|could|can) "
    r"(found|available|provided|located|retrieved))",
    flags=re.IGNORECASE,
)


def _format_icp_block(icp: Dict[str, Any]) -> str:
    """Render the ICP dict in a compact, LLM-friendly block."""
    if not icp:
        return "(no ICP provided)"

    ordered_fields = [
        ("prompt", "Prompt"),
        ("industry", "Industry"),
        ("sub_industry", "Sub-industry"),
        ("target_role_types", "Target role types"),
        ("target_roles", "Target roles"),
        ("target_seniority", "Target seniority"),
        ("employee_count", "Employee count"),
        ("company_stage", "Company stage"),
        ("geography", "Geography"),
        ("country", "Country"),
        ("product_service", "Product / service"),
        ("intent_signals", "Expected intent signals"),
    ]

    lines = []
    for key, label in ordered_fields:
        val = icp.get(key)
        if val in (None, "", [], {}):
            continue
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        lines.append(f"- {label}: {val}")
    return "\n".join(lines) if lines else "(empty ICP)"


def _format_signals_block(signals: List[Dict[str, Any]]) -> str:
    """Render the verified intent signals as a numbered block.

    Each ``signal`` entry should have the shape produced by
    ``gateway/fulfillment/scoring.py`` -> ``intent_signals_detail`` items:
    ``url, description, snippet, date, source, matched_icp_signal``.
    Unscored / zero-score signals are filtered out before rendering so the
    LLM only sees the evidence that actually counted.
    """
    keepers = [
        s for s in (signals or [])
        if float(s.get("after_decay_score") or s.get("raw_score") or 0) > 0
    ]
    if not keepers:
        keepers = list(signals or [])
    if not keepers:
        return "(no intent signals)"

    lines = []
    for i, s in enumerate(keepers, 1):
        desc = (s.get("description") or "").strip()
        snippet = (s.get("snippet") or "").strip()
        date = s.get("date") or "n/a"
        matched = s.get("matched_icp_signal")
        source = s.get("source") or "n/a"
        header = f"{i}. Source: {source}"
        if matched:
            header += f"  |  Matches ICP signal: \"{matched}\""
        header += f"  |  Date: {date}"
        lines.append(header)
        if desc:
            lines.append(f"   Summary: {desc}")
        if snippet:
            # Trim snippet so we don't blow the context window on long pages.
            trimmed = snippet if len(snippet) <= 1500 else snippet[:1500] + "..."
            lines.append(f"   Evidence: {trimmed}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _is_refusal(text: str) -> bool:
    """Detect LLM refusal / meta-commentary outputs we should discard.

    Checks only the opening ~200 characters so a passage that legitimately
    mentions a keyword deep in the prose isn't thrown away.  The regex
    covers the families of refusal openings we've actually observed in
    production: "I need to clarify...", "I appreciate the detailed
    request...", "the search results don't contain...", etc.
    """
    if not text:
        return False
    head = text[:250].lower()
    return bool(_REFUSAL_RE.search(head))


def _clean_passage(text: str) -> str:
    """Post-process the LLM output so it actually conforms to the rules.

    Returns empty string if the output is a refusal — the caller treats
    empty as "don't persist", so refusals are dropped silently rather
    than shown to the client as if they were real intent data.
    """
    if not text:
        return ""

    cleaned = text.strip()
    if (cleaned.startswith("\"") and cleaned.endswith("\"")) or \
       (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()

    # Em/en-dashes -> commas (rules explicitly forbid em dashes).
    cleaned = cleaned.replace("\u2014", ",").replace("\u2013", ",")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^(#+\s*|[-*]\s+)", "", cleaned)

    if _is_refusal(cleaned):
        logger.warning(
            f"Intent details LLM returned a refusal / meta-commentary, "
            f"discarding.  First 200 chars: {cleaned[:200]!r}"
        )
        return ""

    return cleaned


def _get_api_key() -> Optional[str]:
    """Pick up the OpenRouter key already provisioned for fulfillment scoring."""
    return (
        os.getenv("FULFILLMENT_OPENROUTER_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENROUTER_KEY")
    )


async def generate_intent_details_passage(
    icp: Dict[str, Any],
    intent_signals_detail: List[Dict[str, Any]],
) -> str:
    """Generate the client-ready Intent Details paragraph.

    Returns an empty string on any failure — the caller (lifecycle) must
    treat a missing passage as non-fatal and proceed with reward payout.
    The passage is persisted on ``fulfillment_score_consensus.intent_details``.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.warning("generate_intent_details_passage: no OpenRouter API key set")
        return ""

    icp_block = _format_icp_block(icp or {})
    signals_block = _format_signals_block(intent_signals_detail or [])

    prompt = _PROMPT_TEMPLATE.format(
        icp_block=icp_block,
        signals_block=signals_block,
    )

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://leadpoet.ai",
                    "X-Title": "LeadPoet Intent Details",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You write client-ready buying-intent summaries "
                                "strictly from the evidence provided by the user. "
                                "Absolute rules: respond with ONLY the final "
                                "paragraph. Never apologize, never say 'I need to "
                                "clarify', 'I appreciate', 'unfortunately', 'based "
                                "on search results', or any similar meta-commentary. "
                                "Never mention your own limitations or the source "
                                "of the information. Never invent facts beyond the "
                                "provided intent signals. If the provided signals "
                                "are thin, write a shorter paragraph using only "
                                "what is there, but still write a paragraph. "
                                "No preamble, no labels, no markdown, no em dashes, "
                                "no links."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 800,
                },
            )
    except Exception as e:
        logger.warning(f"Intent details LLM call failed: {e}")
        return ""

    if resp.status_code != 200:
        logger.warning(
            f"Intent details LLM returned {resp.status_code}: {resp.text[:300]}"
        )
        return ""

    try:
        data = resp.json()
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        )
    except Exception as e:
        logger.warning(f"Intent details LLM response parse failed: {e}")
        return ""

    return _clean_passage(content)
