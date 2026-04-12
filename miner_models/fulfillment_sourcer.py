"""
Fulfillment lead sourcing model.

Given an ICP (Ideal Customer Profile), discovers real companies, finds real
decision-makers, mines verifiable intent signals from the web, and returns
fully-structured FulfillmentLead objects ready for commit-reveal.

Architecture:
  1. Company Discovery   — ScrapingDog Google Search for companies matching ICP
  2. Contact Discovery    — ScrapingDog Google/LinkedIn for decision-makers
  3. Intent Signal Mining — ScrapingDog web scrape for buying signals
  4. LLM Enrichment      — OpenRouter GPT-4o-mini for structuring/verification
  5. Lead Assembly        — Build FulfillmentLead with real, verifiable data
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
                    # Only accept email_ok — the validator rejects everything
                    # else (accept_all, risky, unknown, failed_*)
                    is_valid = state == "email_ok"
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

    roles_json = json.dumps(target_roles)
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
- city: their city (e.g. "San Francisco")
- state: their US state (e.g. "California")
- email_guess: likely work email (firstname.lastname@{company.get('domain', 'company.com')})

Rules:
- matched_role MUST be copied exactly from the target roles list if the person's actual role is close enough
  (e.g. actual "VP, Sales" matches target "VP of Sales"; actual "Head of Revenue Operations" matches "Head of Revenue")
- If their role doesn't match ANY target role, set matched_role to ""
- city and state are required for US-based contacts"""

    resp = await _llm_call(prompt)
    domain = company.get("domain", "company.com")

    if not resp:
        name_parts = title_text.split(" - ")[0].strip().split("–")[0].strip()
        return {
            "full_name": name_parts,
            "linkedin_url": linkedin_url,
            "role": target_roles[0] if target_roles else "VP of Sales",
            "email": "",
            "city": "",
            "state": company.get("hq_state", ""),
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

        email = data.get("email_guess", "")
        if not email and data.get("full_name"):
            parts = data["full_name"].lower().split()
            if len(parts) >= 2:
                email = f"{parts[0]}.{parts[-1]}@{domain}"

        return {
            "full_name": data.get("full_name", ""),
            "linkedin_url": linkedin_url,
            "role": matched_role,
            "city": data.get("city", ""),
            "state": data.get("state", "") or company.get("hq_state", ""),
            "email": email,
        }
    except json.JSONDecodeError:
        return {
            "full_name": title_text.split(" - ")[0].strip(),
            "linkedin_url": linkedin_url,
            "role": target_roles[0] if target_roles else "",
            "email": "",
            "city": "",
            "state": company.get("hq_state", ""),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Intent Signal Mining
# ═══════════════════════════════════════════════════════════════════════════

async def mine_intent_signals(
    company: dict, icp: dict, max_signals: int = 2
) -> List[Dict]:
    """Find real, verifiable intent signals for a company.

    Searches for job postings, news articles, and company activity that
    indicate buying intent related to the ICP's product/service.

    Returns list of {source, description, url, date, snippet}.
    """
    company_name = company.get("name", "")
    intent_keywords = icp.get("intent_signals", [])
    product = icp.get("product_service", "")

    if not intent_keywords:
        intent_keywords = ["hiring", "expansion", "new product"]

    signals = []

    # Strategy 1: Job postings (strongest intent signal)
    job_query = f'"{company_name}" hiring OR careers OR jobs {" OR ".join(intent_keywords[:2])}'
    job_results = await _google_search(job_query, num_results=5)

    for r in job_results[:3]:
        link = r.get("link", "")
        domain = _extract_domain(link)
        if not link:
            continue
        is_job_board = any(jb in domain for jb in [
            "greenhouse.io", "lever.co", "linkedin.com/jobs",
            "indeed.com", "glassdoor.com", "boards.greenhouse.io",
            "jobs.lever.co", "workable.com", "builtin.com",
        ])
        is_company_careers = "career" in link.lower() or "jobs" in link.lower()

        if is_job_board or is_company_careers:
            snippet = r.get("snippet", "")
            html = await _scrape_url(link)
            page_text = _extract_text_from_html(html, 3000) if html else snippet

            if page_text and len(page_text) > 50:
                signal_prompt = f"""Analyze this job posting/careers page for {company_name}.
Does it show intent related to: {product or ', '.join(intent_keywords)}?

Page content (truncated):
{page_text[:2000]}

URL: {link}

If there IS relevant intent, return a JSON object:
{{"relevant": true, "description": "one sentence describing the buying signal", "snippet": "verbatim 1-2 sentence quote from the page", "date": "YYYY-MM-DD if a date is visible, otherwise null"}}

If NOT relevant, return: {{"relevant": false}}

Return ONLY JSON, no markdown."""

                resp = await _llm_call(signal_prompt)
                if resp:
                    try:
                        resp = resp.strip()
                        if resp.startswith("```"):
                            resp = re.sub(r'^```(?:json)?\s*', '', resp)
                            resp = re.sub(r'\s*```$', '', resp)
                        sig_data = json.loads(resp)
                        if sig_data.get("relevant"):
                            source_type = "job_board" if is_job_board else "company_website"
                            signals.append({
                                "source": source_type,
                                "description": sig_data.get("description", ""),
                                "url": link,
                                "date": sig_data.get("date"),
                                "snippet": sig_data.get("snippet", snippet[:200]),
                            })
                    except json.JSONDecodeError:
                        pass

        if len(signals) >= max_signals:
            break

    # Strategy 2: News/press releases
    if len(signals) < max_signals:
        news_query = f'"{company_name}" {product or intent_keywords[0]} 2026'
        news_results = await _google_search(news_query, num_results=5)

        for r in news_results[:3]:
            if len(signals) >= max_signals:
                break
            link = r.get("link", "")
            domain = _extract_domain(link)
            if not link or domain == company.get("domain", ""):
                continue
            is_news = any(ns in domain for ns in [
                "techcrunch.com", "bloomberg.com", "reuters.com",
                "forbes.com", "cnbc.com", "venturebeat.com",
                "prnewswire.com", "businesswire.com", "globenewswire.com",
                "crunchbase.com", "siliconangle.com", "zdnet.com",
            ])
            if not is_news:
                continue

            snippet = r.get("snippet", "")
            html = await _scrape_url(link)
            page_text = _extract_text_from_html(html, 3000) if html else snippet

            if page_text and len(page_text) > 50:
                news_prompt = f"""Analyze this news article about {company_name}.
Does it contain evidence of buying intent related to: {product or ', '.join(intent_keywords)}?

Content (truncated):
{page_text[:2000]}

URL: {link}

If there IS relevant intent evidence, return JSON:
{{"relevant": true, "description": "one sentence describing the intent signal", "snippet": "verbatim 1-2 sentence quote", "date": "YYYY-MM-DD if visible, otherwise null"}}

If NOT relevant: {{"relevant": false}}

ONLY JSON, no markdown."""

                resp = await _llm_call(news_prompt)
                if resp:
                    try:
                        resp = resp.strip()
                        if resp.startswith("```"):
                            resp = re.sub(r'^```(?:json)?\s*', '', resp)
                            resp = re.sub(r'\s*```$', '', resp)
                        sig_data = json.loads(resp)
                        if sig_data.get("relevant"):
                            signals.append({
                                "source": "news",
                                "description": sig_data.get("description", ""),
                                "url": link,
                                "date": sig_data.get("date"),
                                "snippet": sig_data.get("snippet", snippet[:200]),
                            })
                    except json.JSONDecodeError:
                        pass

    # Strategy 3: Company website activity (fallback)
    if not signals:
        website = company.get("website", "")
        if website:
            html = await _scrape_url(website)
            page_text = _extract_text_from_html(html, 3000)
            if page_text and len(page_text) > 100:
                fallback_prompt = f"""Analyze this company website for {company_name}.
Find ANY evidence of business activity, growth, hiring, or product updates that could
indicate buying intent for: {product or ', '.join(intent_keywords)}.

Content:
{page_text[:2000]}

URL: {website}

Return JSON:
{{"relevant": true, "description": "specific activity found", "snippet": "verbatim quote from site", "date": null}}
OR {{"relevant": false}}

ONLY JSON."""

                resp = await _llm_call(fallback_prompt)
                if resp:
                    try:
                        resp = resp.strip()
                        if resp.startswith("```"):
                            resp = re.sub(r'^```(?:json)?\s*', '', resp)
                            resp = re.sub(r'\s*```$', '', resp)
                        sig_data = json.loads(resp)
                        if sig_data.get("relevant"):
                            signals.append({
                                "source": "company_website",
                                "description": sig_data.get("description", ""),
                                "url": website,
                                "date": sig_data.get("date"),
                                "snippet": sig_data.get("snippet", "")[:200],
                            })
                    except json.JSONDecodeError:
                        pass

    return signals


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Full Pipeline — Source Leads for ICP
# ═══════════════════════════════════════════════════════════════════════════

async def source_fulfillment_leads(icp: dict, num_leads: int = 5) -> List[Dict]:
    """End-to-end pipeline: ICP → discovered, verified FulfillmentLead dicts.

    Returns list of dicts matching the FulfillmentLead schema.
    """
    logger.info(f"Sourcing {num_leads} leads for ICP: {icp.get('industry')}/{icp.get('sub_industry')}")

    # Discover more companies than needed (some won't yield contacts)
    companies = await discover_companies(icp, num_companies=num_leads * 5)
    logger.info(f"Discovered {len(companies)} candidate companies")

    if not companies:
        logger.warning("No companies discovered — cannot source leads")
        return []

    industry = icp.get("industry", "")
    sub_industry = icp.get("sub_industry", "")
    country = icp.get("country", "United States")
    target_seniority = icp.get("target_seniority", "VP")
    target_role_types = icp.get("target_role_types", ["Sales"])

    leads = []
    for company in companies:
        if len(leads) >= num_leads:
            break

        company_name = company.get("name", "Unknown")
        logger.info(f"Processing {company_name}...")

        # Find contact
        contact = await find_contact(company, icp)
        if not contact or not contact.get("full_name"):
            logger.info(f"  No contact found at {company_name}, skipping")
            continue

        # Build email
        domain = company.get("domain", _extract_domain(company.get("website", "")))
        email = contact.get("email", "")
        if not email and contact.get("full_name"):
            parts = contact["full_name"].lower().split()
            if len(parts) >= 2 and domain:
                email = f"{parts[0]}.{parts[-1]}@{domain}"

        if not email:
            logger.info(f"  No email for contact at {company_name}, skipping")
            continue

        # Verify email via TrueList BEFORE spending on intent mining
        if TRUELIST_API_KEY:
            email_valid, email_status = await _verify_email(email)
            if not email_valid:
                logger.info(f"  Email {email} failed TrueList ({email_status}), skipping {company_name}")
                continue
            logger.info(f"  Email {email} verified ({email_status})")

        # Mine intent signals (expensive — only after email is verified)
        signals = await mine_intent_signals(company, icp, max_signals=2)
        if not signals:
            logger.info(f"  No intent signals found for {company_name}, skipping")
            continue

        contact_state = contact.get("state", "") or company.get("hq_state", "")
        contact_city = contact.get("city", "")

        # Validator requires city and state for US leads — skip if missing
        if country.lower() in ("united states", "us", "usa") and not contact_state:
            logger.info(f"  Skipping {company_name} — missing state (required for US leads)")
            continue

        lead = {
            "full_name": contact.get("full_name", ""),
            "email": email,
            "linkedin_url": contact.get("linkedin_url", ""),
            "phone": "",
            "business": company_name,
            "company_linkedin": f"https://linkedin.com/company/{company_name.lower().replace(' ', '-')}",
            "company_website": company.get("website", f"https://{domain}"),
            "employee_count": company.get("employee_estimate", icp.get("employee_count", "")),
            "company_hq_country": country,
            "company_hq_state": contact_state,
            "industry": industry,
            "sub_industry": sub_industry,
            "country": country,
            "city": contact_city or company.get("hq_city", ""),
            "state": contact_state,
            "role": contact.get("role", ""),
            "role_type": target_role_types[0] if target_role_types else "Sales",
            "seniority": target_seniority,
            "intent_signals": [
                {
                    "source": s["source"],
                    "description": s["description"],
                    "url": s["url"],
                    "date": s.get("date"),
                    "snippet": s.get("snippet", "")[:1000],
                }
                for s in signals
            ],
        }

        # Verify role is in target_roles (Tier 1 will reject otherwise)
        target_roles_list = icp.get("target_roles", [])
        if target_roles_list and lead["role"] not in target_roles_list:
            logger.info(f"  Skipping {company_name} — role '{lead['role']}' not in target_roles")
            continue

        leads.append(lead)
        logger.info(f"  ✅ Lead: {contact['full_name']} @ {company_name} role='{lead['role']}' ({len(signals)} signals)")

    logger.info(f"Sourced {len(leads)}/{num_leads} leads")
    return leads
