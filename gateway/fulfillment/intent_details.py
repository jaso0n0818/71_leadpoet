"""
Intent Details passage generator (LLM).

Runs on the gateway once per winning fulfillment lead, after consensus.
Synthesizes the miner's verified intent signals into a single
client-ready paragraph for the "Intent Details" UI column.

Model: perplexity/sonar-pro (via OpenRouter).
Key:   FULFILLMENT_OPENROUTER_API_KEY (already wired into the gateway env).

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
SONAR_MODEL = "perplexity/sonar-pro"
SONAR_TIMEOUT_SECONDS = 60


# The exact wording the user specified.  Adjusting this should be a
# product decision, not a code change, so keep the rules verbatim.
_PROMPT_TEMPLATE = """Add to the Intent Details column using the provided ICP and the identified intent signals for the company. Synthesize these signals into a clear, evidence-based explanation of why the company's observed activity indicates meaningful buying intent and aligns with the priorities, needs, or pain points defined in the ICP.

Rules:
- Only produce the Intent Details passage (no other text).
- Intent Details must be expanded with rich buying-signal context explaining why the signal indicates relevance for the ICP.
- Write Intent Details as a single natural paragraph (no subtitles, bullets, or labels).
- Do not reference "this list," "these candidates," or the source file.
- Do not add links or references in the Intent Details.
- Output must be client-ready for direct use in the UI.
- Ensure intent details are in natural paragraph format, not using any em dashes and not overusing hyphens.
- Do not re-reference the client name in intent details.
- Do not restate ICP facts like country / employee count.
- Ensure all details are factual, evidence-based, and grounded in verifiable sources; do not include claims unless they can be supported by credible source material.
- Do not include links. Output should be client-ready.

Inputs:

ICP:
{icp_block}

Intent Signals:
{signals_block}

Output:

A single polished paragraph for the Intent Details column that clearly explains the relevance and strength of the company's buying intent."""


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


def _clean_passage(text: str) -> str:
    """Post-process the LLM output so it actually conforms to the rules.

    Even with clear instructions, models sometimes emit em-dashes or
    wrap the paragraph in quotes.  This is a belt-and-suspenders clean-up
    that never changes the semantic content.
    """
    if not text:
        return ""

    cleaned = text.strip()
    # Strip wrapping quotes if present.
    if (cleaned.startswith("\"") and cleaned.endswith("\"")) or \
       (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()

    # Replace em/en-dashes with commas (preserves readability).
    cleaned = cleaned.replace("\u2014", ",").replace("\u2013", ",")
    # Collapse runs of whitespace (incl. accidental newlines mid-paragraph).
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Strip any markdown header or list prefix that sometimes sneaks in.
    cleaned = re.sub(r"^(#+\s*|[-*]\s+)", "", cleaned)
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
        async with httpx.AsyncClient(timeout=SONAR_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://leadpoet.ai",
                    "X-Title": "LeadPoet Intent Details",
                },
                json={
                    "model": SONAR_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You write concise, client-ready buying-intent "
                                "summaries. Respond with ONLY the final paragraph, "
                                "no preamble, no labels, no markdown."
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
