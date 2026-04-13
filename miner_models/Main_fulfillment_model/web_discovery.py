"""
Web-based company and contact discovery for the Fulfillment Lead Model.

Functions:
  - discover_companies  — ScrapingDog Google Search for ICP-matching companies
  - find_contact        — LinkedIn search + LLM for role-matched contacts
  - _verify_email       — TrueList email_ok verification
  - _google_search      — ScrapingDog Google Search wrapper
  - _scrape_url         — ScrapingDog web scraping wrapper
  - _llm_call           — OpenRouter LLM call wrapper

Intent signal mining and lead orchestration live in discovery.py.
"""

import asyncio
import json
import logging
import os
import re
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

SCRAPINGDOG_API_KEY = os.getenv("SCRAPINGDOG_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("FULFILLMENT_OPENROUTER_API_KEY", "") or os.getenv("OPENROUTER_KEY", "")
TRUELIST_API_KEY = os.getenv("TRUELIST_API_KEY", "")
_GOOGLE_SEARCH_URL = "https://api.scrapingdog.com/google"
_SCRAPE_URL = "https://api.scrapingdog.com/scrape"
_LINKEDIN_URL = "https://api.scrapingdog.com/linkedin"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_TRUELIST_VERIFY_URL = "https://api.truelist.io/api/v1/verify_inline"


# ═══════════════════════════════════════════════════════════════════════════
# Low-level API helpers
# ═══════════════════════════════════════════════════════════════════════════

async def _google_search(query: str, num_results: int = 10) -> List[Dict]:
    """Search Google via ScrapingDog. Returns list of {title, link, snippet}."""
    if not SCRAPINGDOG_API_KEY:
        logger.warning("SCRAPINGDOG_API_KEY not set — cannot search")
        return []
    async with httpx.AsyncClient(timeout=45) as client:
        for attempt in range(3):
            try:
                resp = await client.get(_GOOGLE_SEARCH_URL, params={
                    "api_key": SCRAPINGDOG_API_KEY,
                    "query": query,
                    "results": num_results,
                    "country": "us",
                })
                if resp.status_code in (429, 502, 503):
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                results = data.get("organic_results", data.get("organic_data", []))
                return [
                    {
                        "title": r.get("title", ""),
                        "link": r.get("link", ""),
                        "snippet": r.get("snippet", ""),
                    }
                    for r in results
                ]
            except Exception as e:
                logger.warning(f"Google search attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(2)
    return []


async def _scrape_url(url: str, dynamic: bool = False) -> str:
    """Scrape a URL via ScrapingDog. Returns raw HTML."""
    if not SCRAPINGDOG_API_KEY:
        return ""
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.get(_SCRAPE_URL, params={
                "api_key": SCRAPINGDOG_API_KEY,
                "url": url,
                "dynamic": "true" if dynamic else "false",
            })
            if resp.status_code == 200:
                return resp.text
        except Exception as e:
            logger.warning(f"Scrape failed for {url[:60]}: {e}")
    return ""


async def _llm_call(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenRouter LLM. Returns raw response text."""
    key = OPENROUTER_API_KEY
    if not key:
        logger.warning("No OpenRouter API key — LLM call skipped")
        return ""
    async with httpx.AsyncClient(timeout=60) as client:
        for attempt in range(2):
            try:
                resp = await client.post(_OPENROUTER_URL, headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://leadpoet.ai",
                }, json={
                    "model": f"openai/{model}",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                })
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                if resp.status_code in (429, 502, 503):
                    await asyncio.sleep(3)
                    continue
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt+1} failed: {e}")
    return ""


async def _verify_email(email: str) -> Tuple[bool, str]:
    """Verify an email via TrueList inline API.

    Returns (is_valid, status).  ``is_valid`` is True only when TrueList
    returns ``email_ok``.  Common statuses:
      - email_ok: mailbox exists
      - accept_all / risky: domain accepts any address (can't confirm)
      - failed_no_mailbox: mailbox does not exist
    """
    if not TRUELIST_API_KEY or not email:
        return False, "no_key"
    async with httpx.AsyncClient(timeout=15) as client:
        for attempt in range(2):
            try:
                resp = await client.post(
                    f"{_TRUELIST_VERIFY_URL}?email={email}",
                    headers={"Authorization": f"Bearer {TRUELIST_API_KEY}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # TrueList wraps results in {"emails": [{...}]}
                    email_data = data
                    if "emails" in data and data["emails"]:
                        email_data = data["emails"][0]
                    state = email_data.get("email_state", "unknown")
                    sub_state = email_data.get("email_sub_state", "")
                    # TrueList puts "deliverable" in email_state and
                    # "email_ok" in email_sub_state — check both fields.
                    is_valid = "email_ok" in (state, sub_state)
                    return is_valid, sub_state or state
                if resp.status_code in (429, 502, 503):
                    await asyncio.sleep(2)
                    continue
            except Exception as e:
                logger.warning(f"TrueList verify failed for {email}: {e}")
                if attempt < 1:
                    await asyncio.sleep(1)
    return False, "error"


def _extract_text_from_html(html: str, max_chars: int = 5000) -> str:
    """Strip HTML tags and extract plain text."""
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_chars]


def _extract_domain(url: str) -> str:
    try:
        h = urlparse(url).hostname or ""
        h = h.lower()
        if h.startswith("www."):
            h = h[4:]
        return h
    except Exception:
        return ""


_GENERIC_EMAIL_PREFIXES = frozenset({
    "info", "admin", "sales", "support", "hello", "contact", "team",
    "hr", "careers", "marketing", "press", "media", "office", "help",
    "billing", "accounts", "noreply", "no-reply", "webmaster",
})

_EMAIL_RE = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}')


async def search_email_google(
    full_name: str, company_name: str, domain: str
) -> str:
    """Search Google for a person's publicly listed work email.

    Tries two queries and extracts any ``@domain`` email from the results.
    Returns the email string or "" if nothing usable is found.
    Generic/catch-all prefixes (info@, sales@, etc.) are ignored.
    """
    if not full_name or not domain:
        return ""

    queries = [
        f'"{full_name}" "@{domain}" email',
        f'"{full_name}" "{company_name}" email',
    ]

    for query in queries:
        results = await _google_search(query, num_results=5)
        for r in results:
            text = f"{r.get('title', '')} {r.get('snippet', '')}"
            emails = _EMAIL_RE.findall(text)
            for email in emails:
                email_lower = email.lower()
                prefix = email_lower.split("@")[0]
                email_domain = email_lower.split("@")[1] if "@" in email_lower else ""
                if prefix in _GENERIC_EMAIL_PREFIXES:
                    continue
                # Prefer exact domain match
                if email_domain == domain.lower():
                    return email_lower
                # Accept subdomain match (e.g. mail.company.com)
                if email_domain.endswith(f".{domain.lower()}"):
                    return email_lower

    return ""


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Company Discovery
# ═══════════════════════════════════════════════════════════════════════════

async def discover_companies(icp: dict, num_companies: int = 10) -> List[Dict]:
    """Find real companies matching the ICP using Google Search.

    Returns list of {name, website, linkedin, description, employee_est}.
    """
    industry = icp.get("industry", "")
    sub_industry = icp.get("sub_industry", "")
    employee_count = icp.get("employee_count", "")
    company_stage = icp.get("company_stage", "")
    country = icp.get("country", "United States")
    product_service = icp.get("product_service", "")

    queries = []
    base = f"{sub_industry or industry} companies"
    if employee_count:
        base += f" {employee_count} employees"
    if company_stage:
        base += f" {company_stage}"
    if country:
        base += f" {country}"

    queries.append(base)
    if product_service:
        queries.append(f"{sub_industry or industry} companies using {product_service} {country}")
    queries.append(f"top {sub_industry or industry} startups {company_stage} {country} 2026")

    all_results = []
    for q in queries[:3]:
        results = await _google_search(q, num_results=10)
        all_results.extend(results)

    seen_domains = set()
    companies_raw = []
    for r in all_results:
        domain = _extract_domain(r.get("link", ""))
        if not domain or domain in seen_domains:
            continue
        if any(skip in domain for skip in [
            "google.com", "wikipedia.org", "linkedin.com", "crunchbase.com",
            "twitter.com", "x.com", "facebook.com", "youtube.com",
            "github.com", "reddit.com", "medium.com", "forbes.com",
            "bloomberg.com", "ycombinator.com", "techcrunch.com",
        ]):
            continue
        seen_domains.add(domain)
        companies_raw.append({
            "name_guess": r.get("title", "").split(" - ")[0].split(" | ")[0].strip(),
            "website": r.get("link", ""),
            "domain": domain,
            "snippet": r.get("snippet", ""),
        })

    if not companies_raw:
        logger.warning("No companies found from Google search")
        return []

    prompt = f"""You are a B2B research assistant. Given these search results about {sub_industry or industry} companies, 
extract a structured list of real companies. For each company provide:
- name: the company name
- website: their website URL
- description: one sentence about what they do
- employee_estimate: estimated employee count range (e.g. "50-200", "200-500")
- hq_city: US city where HQ is located (e.g. "San Francisco")
- hq_state: US state where HQ is located (e.g. "California")

Search results:
{json.dumps(companies_raw[:15], indent=2)}

ICP context: {icp.get('prompt', '')}

Return ONLY a JSON array of objects. No markdown, no explanation.
Return at most {num_companies} companies that best match the ICP.
Only include companies that are real businesses (not articles, directories, or lists)."""

    resp = await _llm_call(prompt)
    if not resp:
        return [{
            "name": c["name_guess"],
            "website": c["website"],
            "domain": c["domain"],
            "description": c["snippet"],
            "employee_estimate": employee_count,
            "hq_state": "",
        } for c in companies_raw[:num_companies]]

    try:
        resp = resp.strip()
        if resp.startswith("```"):
            resp = re.sub(r'^```(?:json)?\s*', '', resp)
            resp = re.sub(r'\s*```$', '', resp)
        companies = json.loads(resp)
        for c in companies:
            c.setdefault("domain", _extract_domain(c.get("website", "")))
        return companies[:num_companies]
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM company response")
        return [{
            "name": c["name_guess"],
            "website": c["website"],
            "domain": c["domain"],
            "description": c["snippet"],
            "employee_estimate": employee_count,
            "hq_state": "",
        } for c in companies_raw[:num_companies]]


# ═══════════════════════════════════════════════════════════════════════════
# Location extraction — mirrors validator Stage 4 logic
# ═══════════════════════════════════════════════════════════════════════════

_US_STATE_ABBR = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia',
}
_US_STATE_NAMES = {v.lower(): v for v in _US_STATE_ABBR.values()}
_US_ABBR_SET = set(_US_STATE_ABBR.keys())

_KNOWN_COUNTRIES = {
    'united states', 'united kingdom', 'canada', 'australia', 'germany',
    'france', 'spain', 'italy', 'netherlands', 'india', 'singapore',
    'japan', 'brazil', 'mexico', 'ireland', 'switzerland', 'sweden',
    'norway', 'denmark', 'belgium', 'austria', 'new zealand', 'israel',
}


def _extract_person_location(title: str, snippet: str) -> Dict[str, str]:
    """Extract a person's city and state from LinkedIn search result text.

    Mirrors the validator's Stage 4 location extraction so the miner
    submits the same city/state the validator will verify.

    Returns {"city": "...", "state": "...", "country": "..."}.
    """
    text = f"{title} {snippet}"

    # Pattern 1: "City, State, United States" at end of snippet
    m = re.search(
        r'([A-Z][a-zA-Z\s\-]+),\s*([A-Z][a-zA-Z\s\-]+),\s*(United States|United Kingdom|Canada|Australia|Germany|France|India)\b',
        text,
    )
    if m:
        city, state_or_region, country = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        if country.lower() == "united states":
            state_full = _US_STATE_NAMES.get(state_or_region.lower(), state_or_region)
            return {"city": city, "state": state_full, "country": country}
        return {"city": city, "state": state_or_region, "country": country}

    # Pattern 2: "City, ST" (US state abbreviation)
    m = re.search(r'([A-Z][a-zA-Z\s\-]+),\s*([A-Z]{2})\b', text)
    if m:
        city, abbr = m.group(1).strip(), m.group(2)
        if abbr in _US_ABBR_SET:
            return {
                "city": city,
                "state": _US_STATE_ABBR[abbr],
                "country": "United States",
            }

    # Pattern 3: "City, Full State Name" (e.g., "Austin, Texas")
    m = re.search(r'([A-Z][a-zA-Z\s\-]+),\s*([A-Z][a-zA-Z\s]+)', text)
    if m:
        city, maybe_state = m.group(1).strip(), m.group(2).strip()
        if maybe_state.lower() in _US_STATE_NAMES:
            return {
                "city": city,
                "state": _US_STATE_NAMES[maybe_state.lower()],
                "country": "United States",
            }

    # Pattern 4: follower count pattern — "City, State. 500+ followers"
    m = re.search(
        r'([A-Z][a-zA-Z\s\-]+(?:,\s*[A-Z][a-zA-Z\s\-]+)*)\.\s*\d+[KMk]?\+?\s*(?:followers?|connections?)',
        text,
    )
    if m:
        loc_text = m.group(1).strip()
        parts = [p.strip() for p in loc_text.split(",")]
        if len(parts) >= 2:
            city = parts[0]
            state_part = parts[1]
            if state_part.lower() in _US_STATE_NAMES:
                return {
                    "city": city,
                    "state": _US_STATE_NAMES[state_part.lower()],
                    "country": "United States",
                }
            if state_part in _US_ABBR_SET:
                return {
                    "city": city,
                    "state": _US_STATE_ABBR[state_part],
                    "country": "United States",
                }
            if state_part.lower() in _KNOWN_COUNTRIES:
                return {"city": city, "state": "", "country": state_part}

    return {"city": "", "state": "", "country": ""}


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Contact Discovery
# ═══════════════════════════════════════════════════════════════════════════

async def find_contact(company: dict, icp: dict) -> Optional[Dict]:
    """Find a real decision-maker at a company matching the ICP's target roles.

    The returned role MUST be one of the ICP's target_roles (exact match)
    because the Tier 1 gate checks ``lead.role in icp.target_roles``.

    Returns {full_name, email, linkedin_url, role, city, state} or None.
    """
    company_name = company.get("name", "")
    target_roles = icp.get("target_roles", [])
    target_seniority = icp.get("target_seniority", "VP")

    if not target_roles:
        target_roles = [f"{target_seniority} of Sales"]

    role_query = " OR ".join(f'"{r}"' for r in target_roles[:3])
    query = f'site:linkedin.com/in/ "{company_name}" ({role_query})'

    results = await _google_search(query, num_results=5)

    linkedin_results = [
        r for r in results
        if "linkedin.com/in/" in r.get("link", "")
    ]

    if not linkedin_results:
        query2 = f'"{company_name}" {target_seniority} sales director linkedin'
        results2 = await _google_search(query2, num_results=5)
        linkedin_results = [
            r for r in results2
            if "linkedin.com/in/" in r.get("link", "")
        ]

    if not linkedin_results:
        return None

    best = linkedin_results[0]
    title_text = best.get("title", "")
    snippet_text = best.get("snippet", "")
    linkedin_url = best.get("link", "").split("?")[0]

    # Extract location from the LinkedIn search result using the same
    # regex patterns the validator uses in Stage 4.  This ensures the
    # miner submits the PERSON's location, not the company HQ.
    person_loc = _extract_person_location(title_text, snippet_text)

    roles_json = json.dumps(target_roles)
    domain = company.get("domain", "company.com")
    prompt = f"""Extract the person's details from this LinkedIn search result and determine
if their role matches any of the target roles.

Title: {title_text}
Snippet: {snippet_text}
LinkedIn URL: {linkedin_url}
Company: {company_name}
Target roles (MUST match one exactly): {roles_json}

Return ONLY a JSON object:
- full_name: their full name
- actual_role: their actual job title from LinkedIn
- matched_role: which target role best matches (MUST be one from the list above, or "" if none match)

Rules:
- matched_role MUST be copied exactly from the target roles list if the person's actual role is close enough
  (e.g. actual "VP, Sales" matches target "VP of Sales"; actual "Head of Revenue Operations" matches "Head of Revenue")
- If their role doesn't match ANY target role, set matched_role to ""
"""

    resp = await _llm_call(prompt)

    if not resp:
        name_parts = title_text.split(" - ")[0].strip().split("–")[0].strip()
        # Search for a publicly listed email
        found_email = await search_email_google(name_parts, company_name, domain)
        return {
            "full_name": name_parts,
            "linkedin_url": linkedin_url,
            "role": target_roles[0] if target_roles else "VP of Sales",
            "email": found_email,
            "city": person_loc.get("city", ""),
            "state": person_loc.get("state", ""),
        }

    try:
        resp = resp.strip()
        if resp.startswith("```"):
            resp = re.sub(r'^```(?:json)?\s*', '', resp)
            resp = re.sub(r'\s*```$', '', resp)
        data = json.loads(resp)

        matched_role = data.get("matched_role", "")
        if not matched_role:
            logger.info(f"  Contact at {company_name} role '{data.get('actual_role')}' doesn't match target roles — skipping")
            return None

        full_name = data.get("full_name", "")
        found_email = await search_email_google(full_name, company_name, domain)

        return {
            "full_name": full_name,
            "linkedin_url": linkedin_url,
            "role": matched_role,
            "city": person_loc.get("city", ""),
            "state": person_loc.get("state", ""),
            "email": found_email,
        }
    except json.JSONDecodeError:
        name_parts = title_text.split(" - ")[0].strip()
        found_email = await search_email_google(name_parts, company_name, domain)
        return {
            "full_name": name_parts,
            "linkedin_url": linkedin_url,
            "role": target_roles[0] if target_roles else "",
            "email": found_email,
            "city": person_loc.get("city", ""),
            "state": person_loc.get("state", ""),
        }
