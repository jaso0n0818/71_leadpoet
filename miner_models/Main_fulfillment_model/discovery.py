"""
Fulfillment Lead Orchestrator — Perplexity-first pipeline.

Uses Perplexity sonar-pro as the FIRST step to discover companies WITH
confirmed intent signals in a single API call. Then finds contacts and
verifies emails only for companies that already have real signals.

Pipeline:
  1. _discover_companies_with_intent — Single Perplexity call finds companies + signals + URLs
  2. find_contact per company        — LinkedIn search for role-matched decision-makers
  3. _verify_email                   — TrueList email_ok verification
  4. _adapt_perplexity_signals       — Convert signals to IntentSignal format
  5. Assemble FulfillmentLead dict

Fallback: If Perplexity returns too few companies, falls back to
Google Search + per-company intent enrichment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from target_fit_model.web_discovery import (
    discover_companies,
    find_contact,
    _verify_email,
    _extract_domain,
    _extract_person_location,
    TRUELIST_API_KEY,
)
from target_fit_model.intent_enrichment import (
    research_company_intent,
    compute_lead_score,
    compute_intent_score_from_signals,
    _cross_source_boost,
)
from target_fit_model.funding_check import detect_funding_intent, check_company_funding, classify_funding_criteria
from target_fit_model.openrouter import chat_completion_json
from target_fit_model.config import PERPLEXITY_MODEL, PERPLEXITY_TIMEOUT
from target_fit_model.scoring import compute_fit_score

# Validator imports — exact same functions the validator uses
from validator_models.stage4_person_verification import (
    run_lead_validation_stage4,
    search_google_async,
    _check_role_via_aimode,
)
from validator_models.stage4_helpers import (
    extract_location_from_text as _s4_extract_location,
    extract_role_from_result as _s4_extract_role,
    get_linkedin_id,
)
from validator_models.stage5_verification import (
    _gse_search_sync,
    _extract_fields_from_results,
    _extract_company_size_from_snippet,
    _check_exact_slug_match,
    _validate_size_match,
    _validate_name_match,
    _parse_hq_to_location,
    _extract_industry_from_snippet,
    _extract_website_from_snippet,
    classify_company_industry,
    _normalize_domain,
)
from validator_models.industry_taxonomy import INDUSTRY_TAXONOMY

import os

SCRAPINGDOG_API_KEY = os.getenv("SCRAPINGDOG_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("FULFILLMENT_OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY", "")

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# IntentSignal Adapter — Perplexity → FulfillmentLead format
# ═══════════════════════════════════════════════════════════════════════════

_DOMAIN_TO_SOURCE = {
    "linkedin.com": "linkedin",
    "greenhouse.io": "job_board",
    "boards.greenhouse.io": "job_board",
    "lever.co": "job_board",
    "jobs.lever.co": "job_board",
    "indeed.com": "job_board",
    "glassdoor.com": "job_board",
    "workable.com": "job_board",
    "builtin.com": "job_board",
    "jobvite.com": "job_board",
    "wellfound.com": "job_board",
    "monster.com": "job_board",
    "ziprecruiter.com": "job_board",
    "techcrunch.com": "news",
    "bloomberg.com": "news",
    "reuters.com": "news",
    "forbes.com": "news",
    "cnbc.com": "news",
    "venturebeat.com": "news",
    "prnewswire.com": "news",
    "businesswire.com": "news",
    "globenewswire.com": "news",
    "crunchbase.com": "news",
    "zdnet.com": "news",
    "siliconangle.com": "news",
    "techradar.com": "news",
    "wired.com": "news",
    "theverge.com": "news",
    "twitter.com": "social_media",
    "x.com": "social_media",
    "facebook.com": "social_media",
    "reddit.com": "social_media",
    "github.com": "github",
    "g2.com": "review_site",
    "capterra.com": "review_site",
    "trustpilot.com": "review_site",
    "wikipedia.org": "wikipedia",
}

_URL_REGEX = re.compile(r'https?://[^\s)<>\]"]+')
_DATE_REGEX = re.compile(
    r'(?:'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{4}'
    r'|\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?'
    r'|\d{1,2}[-/]\d{1,2}[-/]\d{4}'
    r')',
    re.IGNORECASE,
)

_MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def _classify_url(url: str, company_domain: str = "") -> str:
    """Map a URL to an IntentSignalSource value."""
    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
    except Exception:
        return "other"

    for domain_key, source in _DOMAIN_TO_SOURCE.items():
        if domain_key in hostname:
            return source

    if company_domain and company_domain in hostname:
        return "company_website"

    return "other"


def _normalize_date(raw_date: Optional[str]) -> Optional[str]:
    """Normalize a date string to YYYY-MM-DD or return None.

    Handles: "2026-02", "2026-02-15", "2026/3", "March 2026", None, "".
    The FulfillmentLead validator rejects anything not YYYY-MM-DD.
    """
    if not raw_date or not raw_date.strip():
        return None
    d = raw_date.strip()

    # Already YYYY-MM-DD
    if re.match(r'^\d{4}-\d{2}-\d{2}$', d):
        return d

    # YYYY-MM (missing day) — append -01
    if re.match(r'^\d{4}-\d{1,2}$', d):
        parts = d.split("-")
        return f"{parts[0]}-{parts[1].zfill(2)}-01"

    # YYYY/MM/DD or YYYY/MM
    if re.match(r'^\d{4}/\d{1,2}(/\d{1,2})?$', d):
        parts = d.split("/")
        mo = parts[1].zfill(2)
        day = parts[2].zfill(2) if len(parts) > 2 else "01"
        return f"{parts[0]}-{mo}-{day}"

    # "March 2026", "Mar 2026"
    for month_name, month_num in _MONTH_MAP.items():
        if d.lower().startswith(month_name):
            digits = re.findall(r'\d{4}', d)
            if digits:
                return f"{digits[0]}-{month_num}-01"

    return None


def _extract_date_from_text(text: str) -> Optional[str]:
    """Extract the first date from evidence text and return as YYYY-MM-DD."""
    m = _DATE_REGEX.search(text)
    if not m:
        return None
    return _normalize_date(m.group(0).strip().rstrip(","))


def _adapt_perplexity_signals(
    perplexity_result: Dict,
    icp_intent_keywords: List[str],
    company_domain: str = "",
) -> List[Dict]:
    """Convert Perplexity intent research results to IntentSignal-compatible dicts."""
    raw_signals = perplexity_result.get("signals", [])
    adapted = []

    fallback_url = f"https://{company_domain}" if company_domain else ""

    for sig in raw_signals:
        if not sig.get("match"):
            continue

        relevance = sig.get("relevance_score", 0)
        if relevance < 0.3:
            continue

        evidence = sig.get("evidence") or ""
        signal_name = sig.get("signal", "")

        urls = _URL_REGEX.findall(evidence)
        url = urls[0].rstrip(".,;)") if urls else ""

        snippet_text = _URL_REGEX.sub("", evidence).strip()
        snippet_text = re.sub(r'\s+', ' ', snippet_text).strip()
        if not snippet_text:
            snippet_text = signal_name

        source_type = _classify_url(url, company_domain) if url else "other"
        extracted_date = _extract_date_from_text(evidence)

        if not url:
            url = fallback_url
            source_type = "company_website" if company_domain else "other"

        if not url:
            continue

        adapted.append({
            "source": source_type,
            "description": signal_name[:500],
            "url": url,
            "date": extracted_date,
            "snippet": snippet_text[:1000],
        })

    return adapted


def _adapt_direct_signals(
    signals_list: List[Dict],
    company_domain: str = "",
) -> List[Dict]:
    """Adapt signals from the Perplexity-first discovery format.

    Each signal dict has: signal, evidence, url, date (already structured).
    """
    adapted = []
    fallback_url = f"https://{company_domain}" if company_domain else ""

    for sig in signals_list:
        url = (sig.get("url") or "").strip().rstrip(".,;)")
        evidence = sig.get("evidence") or ""
        signal_name = sig.get("signal") or sig.get("description") or ""

        if not url:
            # Try to extract URL from evidence text
            urls = _URL_REGEX.findall(evidence)
            url = urls[0].rstrip(".,;)") if urls else ""

        source_type = _classify_url(url, company_domain) if url else "other"

        snippet_text = _URL_REGEX.sub("", evidence).strip()
        snippet_text = re.sub(r'\s+', ' ', snippet_text).strip()
        if not snippet_text:
            snippet_text = signal_name

        raw_date = sig.get("date") or _extract_date_from_text(evidence)
        extracted_date = _normalize_date(raw_date)

        if not url:
            url = fallback_url
            source_type = "company_website" if company_domain else "other"

        if not url:
            continue

        adapted.append({
            "source": source_type,
            "description": signal_name[:500],
            "url": url,
            "date": extracted_date,
            "snippet": snippet_text[:1000],
        })

    return adapted


def _count_matching_signals(
    adapted_signals: List[Dict],
    icp_keywords: List[str],
) -> int:
    """Count how many ICP intent keywords have at least one matching signal."""
    matched = 0
    for kw in icp_keywords:
        kw_lower = kw.lower()
        for sig in adapted_signals:
            desc = (sig.get("description") or "").lower()
            snippet = (sig.get("snippet") or "").lower()
            if kw_lower in desc or kw_lower in snippet:
                matched += 1
                break
    return matched


# ═══════════════════════════════════════════════════════════════════════════
# Perplexity-First Company + Intent Discovery
# ═══════════════════════════════════════════════════════════════════════════

_PERPLEXITY_SYSTEM = (
    "You are a B2B sales intelligence researcher. You find real companies "
    "with buying signals. Include source URLs when available. "
    "Return ONLY valid JSON — no explanations, no caveats, no markdown."
)


async def _discover_companies_with_intent(
    icp: dict, num_companies: int = 15, exclude_companies: set = None,
) -> List[Dict]:
    """Single Perplexity sonar-pro call to find companies with intent signals.

    Returns list of dicts with:
      name, website, domain, description, employee_estimate,
      hq_city, hq_state, signals: [{signal, evidence, url, date}]
    """
    industry = icp.get("industry", "")
    sub_industry = icp.get("sub_industry", "")
    employee_count = icp.get("employee_count", "")
    country = icp.get("country", "United States")
    intent_keywords = icp.get("intent_signals", [])
    product_service = icp.get("product_service", "")
    prompt_text = icp.get("prompt", "")

    six_months_ago = (datetime.now() - timedelta(days=180)).strftime("%B %Y")

    intent_list = "\n".join(f"- {s}" for s in intent_keywords) if intent_keywords else "- hiring\n- expansion\n- new product launch"

    exclude_text = ""
    if exclude_companies:
        exclude_list = sorted(exclude_companies)[:50]
        exclude_text = f"\n\nDO NOT include these companies (already processed):\n{', '.join(exclude_list)}\n"

    prompt = f"""Find {num_companies} real companies that match the profile below AND show
public evidence of the listed intent signals from the last 6 months (since {six_months_ago}).

COMPANY PROFILE:
- Industry: {industry}{f' / {sub_industry}' if sub_industry else ''}
- Company size: {employee_count or 'any'}
- Country: {country}
{f'- The buyer sells: {product_service}' if product_service else ''}
{f'- Additional context: {prompt_text}' if prompt_text else ''}

INTENT SIGNALS (find evidence of ANY of these):
{intent_list}
{exclude_text}
RULES:
- Only include real, currently operating companies — no fictional or defunct ones
- Each company must have at least 1 signal with real, publicly verifiable evidence
- Include a source URL for each signal when available
- Don't force matches — skip signals that don't genuinely apply
- Find DIFFERENT companies each time — diversify across the industry

Return ONLY a JSON array:
[{{
  "name": "Company Name",
  "website": "https://company.com",
  "description": "One sentence about what they do",
  "employee_estimate": "50-200",
  "hq_city": "San Francisco",
  "hq_state": "California",
  "signals": [{{
    "signal": "exact signal text from list",
    "evidence": "what you found with date",
    "url": "https://source-url",
    "date": "YYYY-MM"
  }}]
}}]

Return at least {num_companies} companies. No explanation text."""

    print(f"  [Perplexity] Searching for {num_companies} companies with intent signals...")

    result = await asyncio.to_thread(
        chat_completion_json,
        prompt=prompt,
        model=PERPLEXITY_MODEL,
        system_prompt=_PERPLEXITY_SYSTEM,
        temperature=0.3,
        max_tokens=8000,
        timeout=PERPLEXITY_TIMEOUT,
    )

    if not result:
        print(f"  [Perplexity] No response (result={result!r}) — falling back to Google Search")
        print(f"  [Perplexity] Prompt length: {len(prompt)} chars, timeout: {PERPLEXITY_TIMEOUT}s")
        return []

    if isinstance(result, dict):
        result = result.get("companies", result.get("results", [result]))

    if not isinstance(result, list):
        print(f"  [Perplexity] Unexpected response format — falling back")
        return []

    companies = []
    for item in result:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "").strip()
        website = item.get("website", "").strip()
        signals = item.get("signals", [])

        if not name:
            continue

        valid_signals = [s for s in signals if isinstance(s, dict)]

        domain = _extract_domain(website) if website else ""

        companies.append({
            "name": name,
            "website": website,
            "domain": domain,
            "description": item.get("description", ""),
            "employee_estimate": item.get("employee_estimate", ""),
            "hq_city": item.get("hq_city", ""),
            "hq_state": item.get("hq_state", ""),
            "signals": valid_signals,
        })

    print(f"  [Perplexity] Found {len(companies)} companies with intent signals")
    for c in companies[:5]:
        sig_count = len(c.get("signals", []))
        print(f"    - {c['name']} ({c.get('domain', '?')}) — {sig_count} signals")

    return companies


# ═══════════════════════════════════════════════════════════════════════════
# Batch Intent Re-check — rescue email-verified leads with no signals
# ═══════════════════════════════════════════════════════════════════════════

async def _batch_intent_recheck(
    pool: List[Dict], icp: dict, max_leads: int = 5
) -> List[Dict]:
    """Single Perplexity call to find intent signals for email-verified leads.

    Takes a pool of partially-built lead dicts (verified email, no intent)
    and asks Perplexity to check if any of them show the requested signals.
    """
    intent_keywords = icp.get("intent_signals", [])
    product_service = icp.get("product_service", "")
    prompt_text = icp.get("prompt", "")

    leads_text = "\n".join(
        f"{i+1}. {p['full_name']}, {p.get('role', '?')} at {p['business']} ({p.get('company_website', '?')})"
        for i, p in enumerate(pool[:20])
    )
    signals_text = "\n".join(f"- {s}" for s in intent_keywords)

    prompt = f"""I have verified leads (person + company). For each one, determine whether
there is REAL, PUBLIC evidence from the last 6 months that connects to any
of the intent signals below.

LEADS:
{leads_text}

INTENT SIGNALS:
{signals_text}

{f'CONTEXT: The buyer sells "{product_service}"' if product_service else ''}
{f'{prompt_text}' if prompt_text else ''}

RULES:
- Evidence must be publicly verifiable (has a URL someone can visit)
- Evidence must be from the last 6 months
- Evidence can be about the COMPANY or the PERSON specifically
- Match each piece of evidence to the most relevant intent signal from the list
- If a signal doesn't apply to a lead, skip it — don't force a match

Return ONLY a JSON array:
[{{"person": "Full Name", "company": "Company Name", "signals": [
  {{"signal": "exact signal text from list", "evidence": "what you found",
   "url": "https://...", "date": "YYYY-MM"}}
]}}]

Omit leads where you found no evidence. No explanation text."""

    print(f"  [Batch Intent] Asking Perplexity about {len(pool)} companies...")

    result = await asyncio.to_thread(
        chat_completion_json,
        prompt=prompt,
        model=PERPLEXITY_MODEL,
        system_prompt=_PERPLEXITY_SYSTEM,
        temperature=0,
        max_tokens=6000,
        timeout=PERPLEXITY_TIMEOUT,
    )

    if not result:
        print(f"  [Batch Intent] No response from Perplexity")
        return []

    if isinstance(result, dict):
        result = result.get("companies", result.get("results", result.get("leads", [])))
    if not isinstance(result, list):
        return []

    # Map results back to pool leads by company name OR person name
    lead_signals = {}
    for item in result:
        if not isinstance(item, dict):
            continue
        company = (item.get("company") or "").strip().lower()
        person = (item.get("person") or "").strip().lower()
        signals = item.get("signals", [])
        if signals:
            if company:
                lead_signals[company] = signals
            if person:
                lead_signals[person] = signals

    print(f"  [Batch Intent] Found signals for {len(lead_signals)} leads")

    rescued = []
    for partial in pool:
        if len(rescued) >= max_leads:
            break

        biz_lower = partial["business"].lower()
        person_lower = partial.get("full_name", "").lower()
        raw_signals = lead_signals.get(biz_lower) or lead_signals.get(person_lower)
        if not raw_signals:
            continue

        domain = partial.get("_domain", _extract_domain(partial.get("company_website", "")))
        adapted = _adapt_direct_signals(raw_signals, company_domain=domain)
        if not adapted:
            continue

        matching = _count_matching_signals(adapted, intent_keywords)
        print(
            f"    Rescued: {partial['full_name']} @ {partial['business']} "
            f"({len(adapted)} signals, {matching}/{len(intent_keywords)} matched)"
        )

        lead = {**partial, "intent_signals": adapted, "_matching_signal_count": matching, "_intent_score": 0.7}
        lead.pop("_domain", None)
        rescued.append(lead)

    return rescued


# ═══════════════════════════════════════════════════════════════════════════
# Validator-Equivalent Verification & Correction
# Uses the EXACT same methods the validator uses in Stage 4 + Stage 5.
# Instead of rejecting on mismatch, CORRECTS the lead data.
# ═══════════════════════════════════════════════════════════════════════════


async def _verify_company_on_linkedin(
    company_name: str,
    company_linkedin: str,
    claimed_employee_count: str = "",
) -> Dict:
    """Search the company LinkedIn page and extract verified fields.

    Uses the exact same Q1 query and extraction functions from Stage 5.
    Returns dict with: slug, name, employee_count, headquarters, industry,
    website, hq_city, hq_state, hq_country, found (bool).
    """
    result = {
        "found": False,
        "slug": "",
        "name": "",
        "employee_count": "",
        "headquarters": "",
        "industry": "",
        "website": "",
        "hq_city": "",
        "hq_state": "",
        "hq_country": "",
    }

    slug = ""
    if company_linkedin:
        m = re.search(r'linkedin\.com/company/([^/?#]+)', company_linkedin.lower())
        if m:
            slug = m.group(1)
    result["slug"] = slug

    if not slug:
        print(f"    [Company LI] No slug from {company_linkedin}")
        return result

    # Q1: Exact Stage 5 query
    q1_query = f'site:linkedin.com/company/{slug} "Industry" "Company size" "Headquarters"'
    print(f"    [Company LI] Q1: {q1_query}")
    q1_result = await asyncio.to_thread(_gse_search_sync, q1_query, 10)

    if q1_result.get("error"):
        print(f"    [Company LI] Q1 error: {q1_result['error']}")
    else:
        extracted = _extract_fields_from_results(q1_result.get("results", []), slug)
        if extracted["exact_slug_found"]:
            result["found"] = True
            result["name"] = extracted.get("title_company_name", "")
            result["employee_count"] = extracted.get("company_size", "")
            result["headquarters"] = extracted.get("headquarters", "")
            result["industry"] = extracted.get("industry", "")
            result["website"] = extracted.get("website", "")

    # Q2 fallback if missing fields
    if result["found"] and not all([result["employee_count"], result["headquarters"]]):
        q2_query = f'{company_name} linkedin company size industry headquarters'
        print(f"    [Company LI] Q2: {q2_query}")
        q2_result = await asyncio.to_thread(_gse_search_sync, q2_query, 10)
        if not q2_result.get("error"):
            extracted = _extract_fields_from_results(q2_result.get("results", []), slug)
            if extracted["exact_slug_found"]:
                if not result["employee_count"] and extracted.get("company_size"):
                    result["employee_count"] = extracted["company_size"]
                if not result["headquarters"] and extracted.get("headquarters"):
                    result["headquarters"] = extracted["headquarters"]
                if not result["industry"] and extracted.get("industry"):
                    result["industry"] = extracted["industry"]
                if not result["website"] and extracted.get("website"):
                    result["website"] = extracted["website"]

    # If slug wasn't found in any query, try searching by company name
    if not result["found"]:
        q_name = f'site:linkedin.com/company/ "{company_name}" "Company size"'
        print(f"    [Company LI] Name search: {q_name}")
        name_result = await asyncio.to_thread(_gse_search_sync, q_name, 5)
        if not name_result.get("error"):
            for r in name_result.get("results", []):
                link = r.get("link", "")
                li_m = re.search(r'linkedin\.com/company/([^/?#]+)', link.lower())
                if li_m:
                    found_slug = li_m.group(1)
                    combined = f"{r.get('title', '')} {r.get('snippet', '')}"
                    name_match, _ = _validate_name_match(company_name, combined.split("|")[0].strip()[:60])
                    if name_match or company_name.lower() in combined.lower():
                        result["found"] = True
                        result["slug"] = found_slug
                        result["employee_count"] = _extract_company_size_from_snippet(combined)
                        result["industry"] = _extract_industry_from_snippet(combined)
                        result["website"] = _extract_website_from_snippet(combined)
                        print(f"    [Company LI] Found via name search: slug={found_slug}")
                        break

    # Parse HQ to structured location
    if result["headquarters"]:
        hq_city, hq_state, hq_country, _ = _parse_hq_to_location(result["headquarters"])
        result["hq_city"] = hq_city
        result["hq_state"] = hq_state
        result["hq_country"] = hq_country

    if result["found"]:
        print(
            f"    [Company LI] Verified: size={result['employee_count']}, "
            f"industry={result['industry']}, HQ={result['headquarters']}"
        )
    else:
        print(f"    [Company LI] Company not found on LinkedIn")

    return result


def _get_valid_industry_pair(
    linkedin_industry: str,
    company_description: str = "",
    icp_industry: str = "",
    icp_sub_industry: str = "",
) -> Tuple[str, str]:
    """Map to a valid (industry, sub_industry) pair from the taxonomy.

    Strategy:
      1. If icp_industry/sub_industry is already a valid taxonomy pair, use it
      2. Try to find a matching sub_industry from the LinkedIn industry
      3. Fall back to the ICP's industry with a best-guess sub_industry
    """
    # Build valid pairs set
    valid_pairs = set()
    for sub, data in INDUSTRY_TAXONOMY.items():
        for ind in data["industries"]:
            valid_pairs.add((ind, sub))

    # Check if ICP pair is already valid
    if icp_industry and icp_sub_industry:
        for ind, sub in valid_pairs:
            if ind.lower() == icp_industry.lower() and sub.lower() == icp_sub_industry.lower():
                return ind, sub

    # Try to find sub_industries matching the LinkedIn industry
    if linkedin_industry:
        li_lower = linkedin_industry.lower()
        matching_subs = []
        for sub, data in INDUSTRY_TAXONOMY.items():
            for ind in data["industries"]:
                if li_lower in ind.lower() or ind.lower() in li_lower:
                    matching_subs.append((ind, sub))
        if matching_subs:
            if company_description:
                desc_lower = company_description.lower()
                scored = []
                for ind, sub in matching_subs:
                    defn = INDUSTRY_TAXONOMY[sub].get("definition", "").lower()
                    overlap = sum(1 for w in sub.lower().split() if w in desc_lower)
                    overlap += sum(1 for w in defn.split() if len(w) > 4 and w in desc_lower)
                    scored.append((overlap, ind, sub))
                scored.sort(reverse=True)
                return scored[0][1], scored[0][2]
            return matching_subs[0]

    # Try ICP industry as parent
    if icp_industry:
        icp_lower = icp_industry.lower()
        for sub, data in INDUSTRY_TAXONOMY.items():
            for ind in data["industries"]:
                if ind.lower() == icp_lower:
                    return ind, sub

    # Last resort: use ICP values even if not in taxonomy (will score lower)
    return icp_industry, icp_sub_industry


async def _find_actual_role(
    linkedin_url: str,
    company_name: str,
    full_name: str,
    search_results: list,
) -> str:
    """Find the person's actual role using the exact validator methods.

    Pipeline (mirrors Stage 4):
      1. extract_role_from_result on all search results
      2. AI Mode: ask ScrapingDog about the LinkedIn profile
      3. LLM fallback on AI Mode prose
    """
    # Step 1: Extract from search results (same as Stage 4 Step 7)
    for r in search_results:
        role = _s4_extract_role(r, full_name, company_name)
        if role:
            print(f"    Found role from search results: {role}")
            return role

    # Step 2: AI Mode — ask for the actual role (same API as Stage 4 Step 7.5)
    if SCRAPINGDOG_API_KEY and linkedin_url:
        print(f"    [AI Mode] Asking for actual role on profile...")
        import time as _time
        import requests as _requests
        from concurrent.futures import ThreadPoolExecutor

        _executor = ThreadPoolExecutor(max_workers=1)
        query = (
            f'For LinkedIn profile {linkedin_url}, what is the current job title '
            f'at "{company_name}"? '
            f'Return JSON only: {{"role":"the exact current job title"}}'
        )

        last_markdown = ""

        def _call():
            nonlocal last_markdown
            for attempt in range(3):
                try:
                    resp = _requests.get(
                        'https://api.scrapingdog.com/google/ai_mode',
                        params={'api_key': SCRAPINGDOG_API_KEY, 'query': query},
                        timeout=90,
                    )
                    if resp.status_code in (502, 503, 429):
                        if attempt < 2:
                            _time.sleep(2 + attempt * 2)
                            continue
                        return None
                    if resp.status_code == 200:
                        data = resp.json()
                        md = data.get('markdown', '') or ''
                        last_markdown = md
                        # Try parsing JSON
                        m = re.search(r'\{[^{}]*"role"[^{}]*\}', md)
                        if m:
                            try:
                                obj = json.loads(m.group())
                                if obj.get("role"):
                                    return obj["role"]
                            except Exception:
                                pass
                        return None
                except Exception:
                    if attempt < 2:
                        _time.sleep(2 + attempt * 2)
                        continue
                    return None
            return None

        loop = asyncio.get_running_loop()
        aimode_role = await loop.run_in_executor(_executor, _call)

        if aimode_role:
            print(f"    [AI Mode] Found role: {aimode_role}")
            return aimode_role

        # Step 3: LLM fallback on AI Mode prose
        if last_markdown and OPENROUTER_API_KEY:
            from target_fit_model.web_discovery import _llm_call
            prompt = (
                f"Extract the current job title for {full_name} at {company_name} "
                f"from this text. Return ONLY the job title, nothing else.\n\n"
                f"Text:\n{last_markdown[:1500]}"
            )
            llm_resp = await _llm_call(prompt)
            if llm_resp and len(llm_resp) < 100 and "sorry" not in llm_resp.lower():
                role = llm_resp.strip().strip('"').strip("'")
                print(f"    [LLM] Extracted role: {role}")
                return role

    return ""


async def _classify_industry_full(
    company_name: str,
    company_description: str,
    linkedin_industry: str,
    icp_industry: str = "",
    icp_sub_industry: str = "",
) -> Tuple[str, str]:
    """Classify industry using the full Stage 5 embedding pipeline.

    Pipeline (exact same as Stage 5):
      1. Validate + refine description (LLM)
      2. Embed refined description (Qwen3-Embedding-8B)
      3. Find top 30 candidates by cosine similarity
      4. LLM ranks top 3

    Falls back to simple taxonomy matching if the pipeline fails.
    """
    if company_description and linkedin_industry:
        try:
            classifications, refined, error = await asyncio.to_thread(
                lambda: asyncio.get_event_loop().run_until_complete(
                    classify_company_industry(
                        miner_description=company_description,
                        extracted_content=f"[Industry: {linkedin_industry}] {company_description}",
                        extracted_industry=linkedin_industry,
                        company_name=company_name,
                        miner_industry=icp_industry,
                        miner_sub_industry=icp_sub_industry,
                    )
                )
            )
        except Exception:
            # classify_company_industry is already async-friendly internally
            # but uses sync calls; just call it directly in a thread
            try:
                from validator_models.stage5_verification import (
                    _load_taxonomy_embeddings,
                    _get_embedding_sync,
                    _find_top_candidates,
                    _format_candidates_for_prompt,
                    _call_llm_sync,
                    _clean_refined_description,
                    _parse_classification_response,
                    VALIDATE_REFINE_PROMPT,
                    CLASSIFY_PROMPT,
                    TOP_K_CANDIDATES,
                )

                def _run_classification():
                    taxonomy = _load_taxonomy_embeddings()
                    if not taxonomy:
                        return [], "", "no_taxonomy"

                    content = f"[Industry: {linkedin_industry}] {company_description}"
                    prompt1 = VALIDATE_REFINE_PROMPT.format(
                        company_name=company_name,
                        miner_description=company_description[:2000],
                        extracted_content=content[:3000],
                    )
                    response1 = _call_llm_sync(prompt1)
                    refined = _clean_refined_description(response1) if response1 else None
                    if not refined:
                        return [], "", "invalid_description"

                    query_emb = _get_embedding_sync(refined)
                    if query_emb is None:
                        return [], refined, "embedding_failed"

                    candidates = _find_top_candidates(query_emb, taxonomy, k=TOP_K_CANDIDATES)
                    if not candidates:
                        return [], refined, "no_candidates"

                    candidates_str, candidates_list = _format_candidates_for_prompt(
                        candidates,
                        miner_industry=icp_industry,
                        miner_sub_industry=icp_sub_industry,
                        valid_pairs=taxonomy.get('valid_pairs'),
                    )
                    desc_for_prompt = f"{refined}. LinkedIn Industry: {linkedin_industry}" if linkedin_industry else refined
                    prompt2 = CLASSIFY_PROMPT.format(
                        refined_description=desc_for_prompt,
                        candidates_list=candidates_str,
                    )
                    response2 = _call_llm_sync(prompt2)
                    classifications = _parse_classification_response(response2, candidates_list) if response2 else []
                    return classifications, refined, ""

                classifications, refined, error = await asyncio.to_thread(_run_classification)
            except Exception as e:
                print(f"    [Industry] Classification pipeline failed: {e}")
                classifications, error = [], str(e)

        if classifications:
            top = classifications[0]
            print(f"    [Industry] Classified: {top['industry']}/{top['sub_industry']}")
            return top["industry"], top["sub_industry"]

    # Fall back to simple taxonomy matching
    return _get_valid_industry_pair(
        linkedin_industry=linkedin_industry,
        company_description=company_description,
        icp_industry=icp_industry,
        icp_sub_industry=icp_sub_industry,
    )


async def _verify_and_correct_lead(lead: dict, icp: dict) -> Optional[Dict]:
    """Run validator-equivalent checks and CORRECT the lead data.

    Uses the exact same verification methods as Stage 4 (person) and
    Stage 5 (company).  Instead of rejecting on mismatch, corrects the
    field to the verified value.

    Returns the corrected lead dict, or None if fundamentally invalid
    (e.g. person doesn't exist on LinkedIn at all).
    """
    company_name = lead["business"]
    domain = lead.get("_domain", _extract_domain(lead.get("company_website", "")))

    print(f"\n    ── Validator-equivalent verification: {lead['full_name']} @ {company_name} ──")

    # ===================================================================
    # STAGE 4: Person verification (LinkedIn URL, name, company, location, role)
    # ===================================================================
    s4_lead = {
        "full_name": lead["full_name"],
        "business": lead["business"],
        "linkedin_url": lead.get("linkedin_url", ""),
        "city": lead.get("city", ""),
        "state": lead.get("state", ""),
        "country": lead.get("country", ""),
        "role": lead.get("role", ""),
        "email": lead.get("email", ""),
    }

    s4_result = await run_lead_validation_stage4(
        s4_lead,
        scrapingdog_api_key=SCRAPINGDOG_API_KEY,
        openrouter_api_key=OPENROUTER_API_KEY,
    )

    if s4_result["passed"]:
        print(f"    ✅ Stage 4 PASSED")
        # Use extracted data to overwrite with verified values
        if s4_result["data"].get("extracted_location"):
            ext_loc = s4_result["data"]["extracted_location"]
            parts = ext_loc.split(",")
            if len(parts) >= 2:
                lead["city"] = parts[0].strip()
                lead["state"] = parts[1].strip()
        if s4_result["data"].get("extracted_role"):
            lead["role"] = s4_result["data"]["extracted_role"]
    else:
        reason = s4_result.get("rejection_reason", {})
        failed = reason.get("failed_fields", [])
        msg = reason.get("message", "")
        print(f"    ⚠️ Stage 4 failed: {msg}")

        # LinkedIn URL not found → unfixable
        if "linkedin" in failed:
            # Try harder: search for the person's LinkedIn directly
            print(f"    Searching for correct LinkedIn URL...")
            q = f'site:linkedin.com/in/ "{lead["full_name"]}" "{company_name}"'
            results, _ = await search_google_async(q, SCRAPINGDOG_API_KEY, max_results=5)
            li_results = [r for r in results if "linkedin.com/in/" in r.get("link", "")]
            if li_results:
                new_url = li_results[0]["link"].split("?")[0]
                print(f"    Found: {new_url} — re-running Stage 4")
                lead["linkedin_url"] = new_url
                s4_lead["linkedin_url"] = new_url
                s4_result = await run_lead_validation_stage4(
                    s4_lead,
                    scrapingdog_api_key=SCRAPINGDOG_API_KEY,
                    openrouter_api_key=OPENROUTER_API_KEY,
                )
                if not s4_result["passed"]:
                    print(f"    ❌ Stage 4 still failed with new URL: {s4_result.get('rejection_reason', {}).get('message', '')}")
                    return None
                print(f"    ✅ Stage 4 PASSED with corrected LinkedIn URL")
            else:
                print(f"    ❌ Cannot find LinkedIn profile — skipping lead")
                return None

        # Name not found → unfixable (wrong person)
        elif "full_name" in failed:
            print(f"    ❌ Name not found on LinkedIn — skipping lead")
            return None

        # Company not found → unfixable (wrong company)
        elif "company" in failed:
            print(f"    ❌ Company not found on LinkedIn — skipping lead")
            return None

        # Location mismatch → CORRECT using extracted data or fresh search
        elif "city" in failed or "state" in failed:
            ext_loc = s4_result["data"].get("extracted_location", "")

            # If Stage 4 didn't extract a location, search the profile directly
            if not ext_loc or "," not in ext_loc:
                for r in s4_result["data"].get("search_results", []):
                    text = f"{r.get('title', '')} {r.get('snippet', '')}"
                    candidate = _s4_extract_location(text)
                    if candidate and "," in candidate:
                        ext_loc = candidate
                        break

            # If still nothing, do a dedicated location search
            if not ext_loc or "," not in ext_loc:
                lid = re.search(r'linkedin\.com/in/([^/?#]+)', lead.get("linkedin_url", ""))
                if lid:
                    slug = lid.group(1)
                    print(f"    Searching for actual location on profile...")
                    loc_q = f'site:linkedin.com/in/{slug} location'
                    loc_results, _ = await search_google_async(loc_q, SCRAPINGDOG_API_KEY, max_results=3)
                    for r in loc_results:
                        text = f"{r.get('title', '')} {r.get('snippet', '')}"
                        candidate = _s4_extract_location(text)
                        if candidate and "," in candidate:
                            ext_loc = candidate
                            break

            if ext_loc and "," in ext_loc:
                parts = ext_loc.split(",")
                old_city, old_state = lead.get("city", ""), lead.get("state", "")
                lead["city"] = parts[0].strip()
                lead["state"] = parts[1].strip()
                lead["company_hq_state"] = lead["state"]
                print(f"    📍 Location CORRECTED: {old_city}, {old_state} → {lead['city']}, {lead['state']}")
            else:
                print(f"    ❌ Could not determine location — skipping lead")
                return None

        # Role mismatch → FIND actual role using exact validator methods
        elif "role" in failed:
            ext_role = s4_result["data"].get("extracted_role", "")

            if not ext_role:
                ext_role = await _find_actual_role(
                    linkedin_url=lead.get("linkedin_url", ""),
                    company_name=company_name,
                    full_name=lead["full_name"],
                    search_results=s4_result["data"].get("search_results", []),
                )

            if ext_role:
                old_role = lead.get("role", "")
                lead["role"] = ext_role
                print(f"    👤 Role CORRECTED: '{old_role}' → '{ext_role}'")
            else:
                print(f"    ❌ Could not determine role — skipping lead")
                return None

        else:
            print(f"    ❌ Stage 4 failed on {failed} — skipping lead")
            return None

    # ===================================================================
    # STAGE 5: Company verification (LinkedIn slug, employee count, HQ, industry)
    # ===================================================================
    company_li = lead.get("company_linkedin", "")
    cli_result = await _verify_company_on_linkedin(
        company_name, company_li,
        claimed_employee_count=lead.get("employee_count", ""),
    )

    if cli_result["found"]:
        # CORRECT company LinkedIn URL if we found a different slug
        if cli_result["slug"] and cli_result["slug"] not in company_li:
            old_li = company_li
            lead["company_linkedin"] = f"https://linkedin.com/company/{cli_result['slug']}"
            print(f"    🔗 Company LinkedIn CORRECTED: {old_li} → {lead['company_linkedin']}")

        # CORRECT employee count to the real LinkedIn value
        if cli_result["employee_count"]:
            old_emp = lead.get("employee_count", "")
            if old_emp != cli_result["employee_count"]:
                lead["employee_count"] = cli_result["employee_count"]
                print(f"    📊 Employee count CORRECTED: '{old_emp}' → '{cli_result['employee_count']}'")

        # CORRECT company website if LinkedIn has it
        if cli_result["website"]:
            li_domain = _normalize_domain(cli_result["website"])
            lead_domain = _normalize_domain(lead.get("company_website", ""))
            if li_domain and li_domain != lead_domain:
                lead["company_website"] = f"https://{li_domain}"
                print(f"    🌐 Website CORRECTED to LinkedIn value: {lead['company_website']}")

        # CORRECT HQ info
        if cli_result["hq_country"]:
            lead["company_hq_country"] = cli_result["hq_country"]
        if cli_result["hq_state"]:
            lead["company_hq_state"] = cli_result["hq_state"]
    else:
        # Company not on LinkedIn — try to find the correct company LinkedIn
        print(f"    Searching for correct company LinkedIn...")
        q = f'site:linkedin.com/company/ "{company_name}"'
        name_r = await asyncio.to_thread(_gse_search_sync, q, 5)
        found_new = False
        for r in name_r.get("results", []):
            link = r.get("link", "")
            m = re.search(r'linkedin\.com/company/([^/?#]+)', link.lower())
            if m and company_name.lower() in f"{r.get('title', '')} {r.get('snippet', '')}".lower():
                new_slug = m.group(1)
                lead["company_linkedin"] = f"https://linkedin.com/company/{new_slug}"
                combined = f"{r.get('title', '')} {r.get('snippet', '')}"
                emp = _extract_company_size_from_snippet(combined)
                if emp:
                    lead["employee_count"] = emp
                found_new = True
                print(f"    🔗 Company LinkedIn FOUND: {lead['company_linkedin']}, size={emp}")
                break
        if not found_new:
            print(f"    ❌ Company not found on LinkedIn — skipping lead")
            return None

    # ===================================================================
    # INDUSTRY / SUB-INDUSTRY: Full Stage 5 classification pipeline
    # ===================================================================
    linkedin_industry = cli_result.get("industry", "") if cli_result["found"] else ""
    company_desc = lead.get("_description", "") or ""

    verified_industry, verified_sub = await _classify_industry_full(
        company_name=company_name,
        company_description=company_desc,
        linkedin_industry=linkedin_industry,
        icp_industry=lead.get("industry", ""),
        icp_sub_industry=lead.get("sub_industry", ""),
    )

    if verified_industry and verified_sub:
        if verified_industry != lead.get("industry") or verified_sub != lead.get("sub_industry"):
            old_ind = f"{lead.get('industry')}/{lead.get('sub_industry')}"
            lead["industry"] = verified_industry
            lead["sub_industry"] = verified_sub
            print(f"    🏭 Industry CORRECTED: {old_ind} → {verified_industry}/{verified_sub}")
    else:
        print(f"    ❌ Could not classify industry — skipping lead")
        return None

    # Clean up internal fields
    lead.pop("_description", None)

    print(f"    ── Verification complete ──")
    return lead


# ═══════════════════════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

MAX_SOURCING_ATTEMPTS = 1000
BATCH_RECHECK_EVERY = 3


async def source_fulfillment_leads(icp: dict, num_leads: int = 5) -> List[Dict]:
    """Perplexity-first pipeline with retry loop.

    Keeps trying until ``num_leads`` are found or ``MAX_SOURCING_ATTEMPTS``
    is reached.  Every ``BATCH_RECHECK_EVERY`` attempts, does a single
    Perplexity batch re-check of all pooled email-verified leads that
    lacked intent signals in earlier attempts.
    """
    industry = icp.get("industry", "")
    sub_industry = icp.get("sub_industry", "")
    intent_keywords = icp.get("intent_signals", [])

    print(f"\n{'='*60}")
    print(f"  FULFILLMENT SOURCER — {industry}/{sub_industry}")
    print(f"  Signals: {intent_keywords}")
    print(f"  Target: {num_leads} leads")
    print(f"{'='*60}")

    all_leads: List[Dict] = []
    email_verified_pool: List[Dict] = []
    seen_companies: set = set()

    consecutive_empty = 0

    for attempt in range(1, MAX_SOURCING_ATTEMPTS + 1):
        if len(all_leads) >= num_leads:
            break

        remaining = num_leads - len(all_leads)

        # Back off when we keep finding nothing new
        if consecutive_empty >= 2:
            pause = min(10 * consecutive_empty, 60)
            print(f"\n  ⏸  Backing off {pause}s ({consecutive_empty} empty attempts)")
            await asyncio.sleep(pause)

        print(f"\n  ── Attempt {attempt}/{MAX_SOURCING_ATTEMPTS} "
              f"(have {len(all_leads)}/{num_leads}, need {remaining} more) ──")

        # Alternate between Perplexity-first (odd) and Google fallback (even)
        if attempt % 2 == 1:
            companies = await _discover_companies_with_intent(
                icp, num_companies=remaining * 5,
                exclude_companies=seen_companies if seen_companies else None,
            )
            use_precomputed = True
        else:
            companies = await discover_companies(icp, num_companies=remaining * 5)
            use_precomputed = False

        # Filter out companies we've already processed
        companies = [
            c for c in companies
            if c.get("name", "").lower() not in seen_companies
        ]

        # When using precomputed signals from Perplexity, skip companies
        # with 0 signals — don't waste API calls on contacts/emails/verification
        if use_precomputed:
            before = len(companies)
            companies = [c for c in companies if c.get("signals")]
            skipped = before - len(companies)
            if skipped:
                print(f"  Filtered out {skipped} companies with 0 intent signals")

        if not companies:
            print(f"  No new companies found this attempt")
            consecutive_empty += 1
        else:
            consecutive_empty = 0
            print(f"  Found {len(companies)} new companies to process")
            new_leads, new_pool = await _process_companies(
                companies, icp, remaining,
                use_precomputed_signals=use_precomputed,
            )

            # Track seen companies
            for c in companies:
                seen_companies.add(c.get("name", "").lower())
            for lead in new_leads:
                seen_companies.add(lead["business"].lower())
            for p in new_pool:
                seen_companies.add(p["business"].lower())

            all_leads.extend(new_leads)
            email_verified_pool.extend(new_pool)

        # Every BATCH_RECHECK_EVERY attempts, batch re-check pooled leads
        if attempt % BATCH_RECHECK_EVERY == 0 and email_verified_pool and len(all_leads) < num_leads:
            remaining = num_leads - len(all_leads)
            lead_companies = {l["business"].lower() for l in all_leads}
            pool = [p for p in email_verified_pool if p["business"].lower() not in lead_companies]

            if pool:
                print(f"\n  [Batch Intent] Re-checking {len(pool)} email-verified leads...")
                rescued = await _batch_intent_recheck(pool, icp, max_leads=remaining)
                all_leads.extend(rescued)
                # Remove rescued from pool
                rescued_names = {r["business"].lower() for r in rescued}
                email_verified_pool = [p for p in email_verified_pool if p["business"].lower() not in rescued_names]

    # Final batch re-check if we still need more
    if len(all_leads) < num_leads and email_verified_pool:
        remaining = num_leads - len(all_leads)
        lead_companies = {l["business"].lower() for l in all_leads}
        pool = [p for p in email_verified_pool if p["business"].lower() not in lead_companies]

        if pool:
            print(f"\n  [Final Batch] Re-checking {len(pool)} remaining pooled leads...")
            rescued = await _batch_intent_recheck(pool, icp, max_leads=remaining)
            all_leads.extend(rescued)

    # Sort by combined lead score (intent 60% + ICP fit 40%)
    all_leads.sort(key=lambda x: x.get("_lead_score", 0), reverse=True)

    for lead in all_leads:
        lead.pop("_matching_signal_count", None)
        lead.pop("_intent_score", None)
        lead.pop("_fit_score", None)
        lead.pop("_lead_score", None)

    print(f"\n{'='*60}")
    print(f"  RESULT: Sourced {len(all_leads)}/{num_leads} leads "
          f"({len(email_verified_pool)} in pool)")
    print(f"{'='*60}\n")

    return all_leads[:num_leads]


async def _process_companies(
    companies: List[Dict],
    icp: dict,
    num_leads: int,
    use_precomputed_signals: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """Process companies into FulfillmentLead dicts.

    Pipeline order:
      1. find_contact (Google/LinkedIn) — real person, role, location, email
      2. TrueList verify (FREE) — confirm email deliverability
      3. _verify_and_correct_lead — Stage 4 + Stage 5 validator checks, correct data
      4. Attach intent signals (precomputed from Perplexity, or fetch individually)
      5. Assemble final lead

    Returns (leads, email_verified_pool).
    """
    industry = icp.get("industry", "")
    sub_industry = icp.get("sub_industry", "")
    country = icp.get("country", "United States")
    target_seniority = icp.get("target_seniority", "VP")
    target_role_types = icp.get("target_role_types", ["Sales"])
    intent_keywords = icp.get("intent_signals", [])
    product_service = icp.get("product_service", "")
    prompt_text = icp.get("prompt", "")
    intent_signals_str = ", ".join(intent_keywords) if intent_keywords else ""

    # Detect if ICP signals mention funding — if so, check each company's funding
    _has_funding_intent = detect_funding_intent(intent_signals_str)
    _funding_criteria = None
    if _has_funding_intent:
        _funding_criteria = await asyncio.to_thread(classify_funding_criteria, intent_signals_str)
        if _funding_criteria:
            print(f"  [Funding] Detected funding intent: {_funding_criteria.get('description', '')}")

    leads = []
    email_verified_pool = []
    for company in companies:
        if len(leads) >= num_leads:
            break

        company_name = company.get("name", "Unknown")
        domain = company.get("domain", _extract_domain(company.get("website", "")))
        print(f"\n  Processing: {company_name} ({domain})")

        # ── Step 1: Find a real contact via Google/LinkedIn ──
        print(f"    Searching Google/LinkedIn for a contact...")
        contact = await find_contact(company, icp)

        if not contact or not contact.get("full_name"):
            print(f"    No matching contact found on Google/LinkedIn, skipping")
            continue

        verified_name = contact["full_name"]
        verified_role = contact.get("role", "")
        verified_linkedin = contact.get("linkedin_url", "")
        verified_city = contact.get("city", "")
        verified_state = contact.get("state", "")
        email = contact.get("email", "")

        print(f"    Found: {verified_name}, {verified_role}")
        if verified_city or verified_state:
            print(f"    Location: {verified_city}, {verified_state}")

        if not email:
            print(f"    No public email found, skipping")
            continue

        print(f"    Email found: {email}")

        # ── Step 2: TrueList verify (FREE) ──
        if TRUELIST_API_KEY:
            email_valid, email_status = await _verify_email(email)
            if not email_valid:
                print(f"    Email {email} failed TrueList ({email_status}), skipping")
                continue
            print(f"    Email verified ({email_status})")
        else:
            print(f"    No TrueList key — using unverified email")

        # ── Build partial lead ──
        if not verified_state and not verified_city:
            verified_state = company.get("hq_state", "")
            verified_city = company.get("hq_city", "")

        discovered_emp_count = company.get("employee_estimate", "") or company.get("employee_count", "")

        company_li = company.get("linkedin", "")
        if not company_li and domain:
            slug = domain.split(".")[0]
            company_li = f"https://linkedin.com/company/{slug}"
        if not company_li:
            company_li = f"https://linkedin.com/company/{company_name.lower().replace(' ', '-')}"

        partial_lead = {
            "full_name": verified_name,
            "email": email,
            "linkedin_url": verified_linkedin,
            "phone": "",
            "business": company_name,
            "company_linkedin": company_li,
            "company_website": company.get("website", f"https://{domain}" if domain else ""),
            "employee_count": discovered_emp_count,
            "company_hq_country": country,
            "company_hq_state": verified_state,
            "industry": industry,
            "sub_industry": sub_industry,
            "country": country,
            "city": verified_city,
            "state": verified_state,
            "role": verified_role,
            "role_type": target_role_types[0] if target_role_types else "Sales",
            "seniority": target_seniority,
            "_domain": domain,
            "_description": company.get("description", ""),
        }

        # ── Step 3: Validator-equivalent verification & correction ──
        corrected = await _verify_and_correct_lead(partial_lead, icp)
        if corrected is None:
            print(f"    ❌ Lead failed verification — skipping")
            continue
        partial_lead = corrected

        # ── Step 4: Get intent signals ──
        raw_perplexity_signals = []  # keep raw format for scoring

        if use_precomputed_signals:
            raw_signals = company.get("signals", [])
            adapted_signals = _adapt_direct_signals(raw_signals, company_domain=domain)
            # Convert precomputed signals to raw format for compute_intent_score_from_signals
            raw_perplexity_signals = [
                {"signal": s.get("signal", s.get("description", "")),
                 "match": True,
                 "relevance_score": 0.8,
                 "evidence": f"{s.get('evidence', '')} {s.get('url', '')}"}
                for s in raw_signals if isinstance(s, dict)
            ]
        else:
            print(f"    Researching intent signals...")
            company_for_intent = {
                "company_name": company_name,
                "website": partial_lead.get("company_website", f"https://{domain}" if domain else ""),
                "company_linkedin": partial_lead.get("company_linkedin", ""),
                "industry": partial_lead.get("industry", industry),
                "description": company.get("description", ""),
            }

            # If ICP mentions funding, check funding first and pass as context
            if _has_funding_intent:
                funded, funding_evidence, _ = await asyncio.to_thread(
                    check_company_funding,
                    company_name,
                    partial_lead.get("company_website", ""),
                    _funding_criteria,
                )
                if funded:
                    company_for_intent["_funding_evidence"] = funding_evidence
                    print(f"    Funding: {funding_evidence[:80]}")

            perplexity_result = await asyncio.to_thread(
                research_company_intent,
                company=company_for_intent,
                product_description=product_service or prompt_text,
                request_description=prompt_text or f"{industry} company, {sub_industry}",
                intent_signals=intent_signals_str,
            )

            if not perplexity_result or not perplexity_result.get("signals"):
                print(f"    No intent signals (saved to pool for batch re-check)")
                email_verified_pool.append(partial_lead)
                continue

            raw_perplexity_signals = perplexity_result.get("signals", [])
            adapted_signals = _adapt_perplexity_signals(
                perplexity_result, intent_keywords, company_domain=domain,
            )

        if not adapted_signals:
            print(f"    No usable intent signals (saved to pool for batch re-check)")
            email_verified_pool.append(partial_lead)
            continue

        matching_count = _count_matching_signals(adapted_signals, intent_keywords)

        # ── Step 5: Proper intent + fit scoring ──
        # Use the founding engineer's tiered scoring (user signals 2x, coverage bonus)
        intent_score = compute_intent_score_from_signals(
            raw_perplexity_signals, original_signals=intent_keywords,
        )

        # Cross-source boost: reward evidence from multiple independent sources
        intent_score, source_count = _cross_source_boost(
            raw_perplexity_signals, intent_score,
        )

        # ICP fit scoring (multi-dimensional: industry, role, location, size, quality)
        fit_score, fit_breakdown = compute_fit_score(partial_lead, icp)

        # Combined score: intent 60% + fit 40%
        lead_score = compute_lead_score(intent_score, fit_score)

        print(
            f"    Intent: {len(adapted_signals)} signals, "
            f"{matching_count}/{len(intent_keywords)} ICP keywords matched, "
            f"score={intent_score:.2f} ({source_count} sources)"
        )
        print(
            f"    Fit: {fit_score:.2f} (ind={fit_breakdown.get('industry_match', 0):.1f}, "
            f"role={fit_breakdown.get('role_match', 0):.1f}, "
            f"loc={fit_breakdown.get('location_match', 0):.1f}, "
            f"size={fit_breakdown.get('size_match', 0):.1f}) "
            f"| Lead score: {lead_score:.3f}"
        )

        lead = {
            **partial_lead,
            "intent_signals": adapted_signals,
            "_matching_signal_count": matching_count,
            "_intent_score": intent_score,
            "_fit_score": fit_score,
            "_lead_score": lead_score,
        }
        lead.pop("_domain", None)

        leads.append(lead)
        print(
            f"    ✅ Lead: {partial_lead['full_name']} @ {company_name} "
            f"role='{partial_lead['role']}' ({len(adapted_signals)} signals, score={lead_score:.3f})"
        )

    return leads, email_verified_pool
