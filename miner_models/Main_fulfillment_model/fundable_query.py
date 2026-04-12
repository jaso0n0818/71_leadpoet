"""
Fundable API query module.

Discovers funded companies via Fundable API, fetches decision-maker leads,
and normalizes locations.

Entry point: discover_fundable_leads()
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from geo_normalize import (
    normalize_location as _geo_normalize,
    infer_country_from_state,
    normalize_country,
)

logger = logging.getLogger(__name__)

# ============================================================
# Config
# ============================================================

FUNDABLE_API_KEY = os.environ.get("TRYFUNDABLE_API_KEY", "")
SCRAPINGDOG_API_KEY = os.environ.get("SCRAPINGDOG_API_KEY", "")
BASE_URL = "https://www.tryfundable.ai/api/v1"
HEADERS = {"Authorization": f"Bearer {FUNDABLE_API_KEY}", "Accept": "application/json"}

# Load mapping file (taxonomy -> Fundable slugs)
_mapping_path = Path(__file__).parent / "fundable_mapping.json"
with open(_mapping_path, encoding="utf-8") as _f:
    FUNDABLE_MAP = json.load(_f)

# Load area-city mappings (186 metro areas)
_area_path = Path(__file__).parent / "area_city_mappings.json"
with open(_area_path, encoding="utf-8") as _f:
    _AREA_MAPPINGS = _f.read()
_AREA_MAPPINGS = json.loads(_AREA_MAPPINGS).get("mappings", {})

# Load geo data for city->state fallback
_geo_path = Path(__file__).parent / "geo_lookup_fast.json"
with open(_geo_path, encoding="utf-8") as _f:
    _geo_raw = json.load(_f)
_INTL_CITIES = {co: set(cities) for co, cities in _geo_raw.get("cities", {}).items()}
del _geo_raw


# ============================================================
# Area normalization
# ============================================================

def _norm_area(a: str) -> str:
    a = a.lower().strip()
    a = re.sub(
        r"\s*(metropolitan area|metropolitan|metroplex|metro area|metro|bay area|area|region)$",
        "", a, flags=re.I,
    )
    a = a.replace("greater ", "").strip()
    return a


_AREA_LOOKUP = {_norm_area(k): v for k, v in _AREA_MAPPINGS.items()}
_CITY_TO_AREA = {}
for _info in _AREA_MAPPINGS.values():
    for _c in _info.get("cities", []):
        _CITY_TO_AREA[_c.lower()] = (_info.get("state", ""), _info.get("country", ""))


# ============================================================
# Location normalization
# ============================================================

def normalize_location(city_raw: str) -> Tuple[str, str, str]:
    """Parse person city field into (city, state, country).
    Uses area_city_mappings.json + geo_lookup_fast.json.
    """
    if not city_raw:
        return "", "", ""
    city_raw = city_raw.strip()

    # 1. Check area name match
    area_norm = _norm_area(city_raw)
    if area_norm in _AREA_LOOKUP:
        info = _AREA_LOOKUP[area_norm]
        main_city = re.sub(r"^(Metro|Greater)\s+", "", city_raw, flags=re.I).strip()
        main_city = re.sub(
            r"\s*(Metropolitan Area|Metropolitan|Metroplex|Metro Area|Metro|Bay Area|Area|Region)$",
            "", main_city, flags=re.I,
        ).strip()
        if "-" in main_city:
            main_city = main_city.split("-")[0].strip()
        return main_city, info.get("state", ""), info.get("country", "")

    # 2. Strip metro/area prefixes and suffixes
    cleaned = re.sub(r"^(Metro|Greater)\s+", "", city_raw, flags=re.I).strip()
    cleaned = re.sub(
        r"\s*(Metropolitan Area|Metropolitan|Metroplex|Metro Area|Metro|Bay Area|Area|Region)$",
        "", cleaned, flags=re.I,
    ).strip()

    # 3. Hyphenated metros — take first city
    if "-" in cleaned and "," not in cleaned:
        first_city = cleaned.split("-")[0].strip()
        if first_city.lower() in _CITY_TO_AREA:
            cleaned = first_city

    # 4. Split by comma
    parts = [x.strip() for x in cleaned.split(",")]
    raw_city = parts[0] if len(parts) >= 1 else ""
    raw_state = parts[1] if len(parts) >= 2 else ""
    raw_country = parts[2] if len(parts) >= 3 else ""

    # 5. Single city — area reverse lookup
    if not raw_state and not raw_country:
        city_lower = raw_city.lower()
        if city_lower in _CITY_TO_AREA:
            state, country = _CITY_TO_AREA[city_lower]
            return raw_city.title(), state, country
        for co, city_set in _INTL_CITIES.items():
            if city_lower in city_set:
                return raw_city.title(), "", co.title()

    # 6. Standard — geo_normalize
    city, state, country = _geo_normalize(raw_city, raw_state, raw_country)
    if country:
        country = normalize_country(country)
    if not country and state:
        country = infer_country_from_state(state)
    return city, state, country


# ============================================================
# Taxonomy → Fundable slug conversion
# ============================================================

def to_industry_slugs(names: List[str]) -> str:
    """Convert taxonomy industry names to comma-separated Fundable slugs."""
    slugs = []
    for name in names:
        name = name.strip()
        slug = FUNDABLE_MAP["industries"].get(name)
        if slug:
            slugs.append(slug)
        else:
            logger.warning(f"Industry '{name}' not found in fundable_mapping.json")
    return ",".join(slugs)


def to_employee_slugs(sizes: List[str]) -> str:
    """Convert taxonomy employee sizes to comma-separated Fundable slugs."""
    slugs = []
    for size in sizes:
        size = size.strip()
        mapped = FUNDABLE_MAP["employee_counts"].get(size)
        if mapped:
            for s in mapped.split(","):
                if s not in slugs:
                    slugs.append(s)
        else:
            logger.warning(f"Employee count '{size}' not found in fundable_mapping.json")
    return ",".join(slugs)


# ============================================================
# Fundable API functions
# ============================================================

def resolve_company(domain_or_url: str) -> Optional[Dict]:
    """Lookup company by domain or LinkedIn URL."""
    url = domain_or_url.strip()
    if "linkedin.com" in url:
        param = {"linkedin": url}
    else:
        domain = url.replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")
        param = {"domain": domain}
    resp = requests.get(f"{BASE_URL}/company/", headers=HEADERS, params=param, timeout=30, allow_redirects=True)
    if resp.status_code != 200:
        return None
    raw = resp.json()
    inner = raw.get("data", {})
    return inner.get("company", inner) if isinstance(inner, dict) else inner


def get_people(company_id: str, page_size: int = 20) -> List[Dict]:
    """Get decision makers (founder, ceo, key_person) with email."""
    resp = requests.get(
        f"{BASE_URL}/people/",
        headers=HEADERS,
        params={
            "company_ids": company_id,
            "roles": "founder,ceo,key_person",
            "contact_types": "email",
            "page_size": page_size,
        },
        timeout=30,
        allow_redirects=True,
    )
    if resp.status_code != 200:
        return []
    return resp.json().get("data", {}).get("people", [])


def resolve_investor_names(investor_ids: List[str]) -> List[str]:
    """Resolve investor UUIDs to names via /investor endpoint."""
    names = []
    for inv_id in investor_ids:
        try:
            resp = requests.get(
                f"{BASE_URL}/investor/",
                headers=HEADERS,
                params={"id": inv_id},
                timeout=30,
                allow_redirects=True,
            )
            if resp.status_code == 200:
                inv = resp.json().get("data", {}).get("investor", {})
                if inv.get("name"):
                    names.append(inv["name"])
        except Exception:
            pass
    return names


# ============================================================
# Data extraction helpers
# ============================================================

def extract_company_data(company: Dict, domain: str = "") -> Dict:
    """Extract normalized company fields from Fundable API response."""
    website = f"https://{domain}" if domain else ""
    industries_raw = company.get("industries", [])
    loc = company.get("location", {}) or {}
    deal = company.get("latest_deal", {}) or {}
    deal_date_raw = deal.get("date", "")
    deal_date = deal_date_raw[:10] if deal_date_raw else ""
    days_since = ""
    if deal_date:
        try:
            days_since = (datetime.now() - datetime.strptime(deal_date, "%Y-%m-%d")).days
        except Exception:
            pass
    deal_amount = ""
    for fin in deal.get("financings", []):
        if fin.get("size_usd"):
            deal_amount = fin["size_usd"]
            break

    return {
        "company_name": company.get("name", ""),
        "website": website,
        "company_linkedin": company.get("linkedin", ""),
        "employees": company.get("num_employees", ""),
        "description": company.get("short_description", ""),
        "industry": industries_raw[0].get("name", "") if industries_raw else "",
        "sub_industry": industries_raw[1].get("name", "") if len(industries_raw) > 1 else "",
        "all_industries": ",".join(i.get("name", "") for i in industries_raw),
        "hq_city": (loc.get("city") or {}).get("name", ""),
        "hq_state": (loc.get("state") or {}).get("name", ""),
        "hq_country": (loc.get("country") or {}).get("name", ""),
        "total_raised": company.get("total_raised"),
        "num_rounds": company.get("num_funding_rounds"),
        "deal_type": deal.get("type", ""),
        "deal_date": deal_date,
        "days_since": days_since,
        "deal_amount": deal_amount,
        "deal_investor_ids": deal.get("investors", []),
    }


def _normalize_to_lead_schema(person: Dict, cd: Dict, investors_str: str) -> Dict:
    """Map Fundable person + company data to the standard lead dict schema
    expected by process_companies() in ui.py."""
    name = person.get("name", "")
    name_parts = name.split(" ", 1)
    first_name = name_parts[0] if name_parts else ""
    last_name = name_parts[1] if len(name_parts) > 1 else ""

    contact_city, contact_state, contact_country = normalize_location(person.get("city", ""))

    return {
        "lead_id": f"fundable_{person.get('id', '')}",
        "first_name": first_name,
        "last_name": last_name,
        "email": person.get("email", ""),
        "role": person.get("title", ""),
        "company_name": cd["company_name"],
        "linkedin": person.get("linkedin_url", ""),
        "website": cd["website"],
        "company_linkedin": cd["company_linkedin"],
        "phone": person.get("phone", ""),
        "industry": cd["industry"],
        "sub_industry": cd["sub_industry"],
        "all_industries": cd.get("all_industries", ""),
        "city": contact_city,
        "state": contact_state,
        "country": contact_country,
        "hq_city": cd["hq_city"],
        "hq_state": cd["hq_state"],
        "hq_country": cd["hq_country"],
        "employee_count": cd["employees"],
        "description": cd["description"],
        "rep_score": 25.0,  # Neutral score for Fundable leads
        "lead_blob": {},
        # Funding-specific fields
        "_funding_type": cd["deal_type"],
        "_funding_date": cd["deal_date"],
        "_funding_amount": cd["deal_amount"],
        "_funding_investors": investors_str,
        "_total_raised": cd["total_raised"],
        "_days_since_funding": cd["days_since"],
        "_funding_rounds": cd["num_rounds"],
        "_funding_evidence": (
            f"Last funding: {cd['deal_type']} on {cd['deal_date']}, "
            f"Total raised: {cd['total_raised']}, "
            f"Investors: {investors_str}"
        ),
    }


# ============================================================
# Main entry point
# ============================================================

def discover_fundable_leads(
    industries: List[str],
    employee_counts: List[str],
    countries: List[str],
    states: List[str],
    cities: List[str],
    funding_days: Optional[int] = None,
    financing_types: str = "",
    max_companies: int = 25,
    progress_callback=None,
) -> List[Dict]:
    """
    Discover funded companies via Fundable API and return leads.

    Returns list of company dicts in the same shape as discover_phases() output:
    [
        {
            "company_name": str, "location": str, "website": str,
            "company_linkedin": str, "industry": str, "sub_industry": str,
            "employee_count": str, "description": str, "unique_roles": [str],
            "fit_score": 0.0, "lead_count": int, "leads": [lead_dict, ...],
            # Funding extras
            "_funding_type": str, "_funding_date": str, ...
        }
    ]
    """
    # Build API params
    params = {
        "roles": "founder,ceo,key_person",
        "contact_types": "email",
        "page_size": 100,
    }

    industry_slugs = to_industry_slugs(industries) if industries else ""
    employee_slugs = to_employee_slugs(employee_counts) if employee_counts else ""

    if funding_days:
        params["deal_start_date"] = (datetime.now() - timedelta(days=int(funding_days))).strftime("%Y-%m-%d")
    if employee_slugs:
        params["employee_count"] = employee_slugs
    if industry_slugs:
        params["industries"] = industry_slugs
    if financing_types:
        params["financing_types"] = financing_types

    # Person location filters (client-side)
    filter_cities = [c.strip().lower() for c in cities if c.strip()]
    filter_states = [s.strip().lower() for s in states if s.strip()]
    filter_countries = [c.strip().lower() for c in countries if c.strip()]
    has_location_filter = bool(filter_cities or filter_states or filter_countries)

    logger.info(f"[Fundable] Searching with params: {params}")
    if has_location_filter:
        logger.info(f"[Fundable] Location filter: cities={filter_cities}, states={filter_states}, countries={filter_countries}")

    # Paginate and process companies
    companies_result = []
    seen_company_ids = set()
    companies_kept = 0
    companies_checked = 0
    page = 0
    api_total = None

    while companies_kept < max_companies:
        params["page"] = page
        try:
            resp = requests.get(f"{BASE_URL}/people/", headers=HEADERS, params=params, timeout=30, allow_redirects=True)
        except Exception as e:
            logger.error(f"[Fundable] API request failed: {e}")
            break

        if resp.status_code != 200:
            logger.error(f"[Fundable] API error {resp.status_code}: {resp.text[:300]}")
            break

        data = resp.json()
        people_batch = data.get("data", {}).get("people", [])
        api_total = data.get("meta", {}).get("total_count", 0)

        if not people_batch:
            logger.info(f"[Fundable] No more results (total: {api_total})")
            break

        # Extract unique companies from this page
        page_companies = {}
        for p in people_batch:
            c = p.get("company", {})
            cid = c.get("id", "")
            if cid and cid not in seen_company_ids and cid not in page_companies:
                page_companies[cid] = c

        for cid, basic in page_companies.items():
            if companies_kept >= max_companies:
                break

            seen_company_ids.add(cid)
            domain = basic.get("domain", "")
            companies_checked += 1
            company_name = basic.get("name", "?")

            if progress_callback:
                progress_callback(f"Checking {company_name} ({companies_kept}/{max_companies})...")

            # Get leads first (free — check location before expensive ops)
            people = get_people(cid)
            if not people:
                continue

            # Location filter
            if has_location_filter:
                matched_people = []
                for p in people:
                    p_city, p_state, p_country = normalize_location(p.get("city", ""))
                    city_ok = not filter_cities or any(fc in p_city.lower() for fc in filter_cities)
                    state_ok = not filter_states or any(fs in p_state.lower() for fs in filter_states)
                    country_ok = not filter_countries or any(fc in p_country.lower() for fc in filter_countries)
                    if city_ok and state_ok and country_ok:
                        matched_people.append(p)
                if not matched_people:
                    continue
                people = matched_people

            # Get full company details
            company = resolve_company(domain) if domain else None
            if not company:
                continue

            cd = extract_company_data(company, domain)

            # Skip ScrapingDog checks — trust Fundable data
            # HQ and employee size come directly from Fundable API

            # Resolve investor names
            inv_ids = cd["deal_investor_ids"]
            investors_str = ""
            if inv_ids and isinstance(inv_ids[0], str) and len(inv_ids[0]) > 20:
                investors_str = ", ".join(resolve_investor_names(inv_ids))

            # Build leads in standard schema
            leads = [_normalize_to_lead_schema(p, cd, investors_str) for p in people]

            # Build company dict in same shape as discover_phases() output
            unique_roles = list({l["role"] for l in leads if l.get("role")})
            company_entry = {
                "company_name": cd["company_name"],
                "location": f"{cd['hq_city']}, {cd['hq_state']}, {cd['hq_country']}".strip(", "),
                "website": cd["website"],
                "company_linkedin": cd["company_linkedin"],
                "industry": cd["industry"],
                "sub_industry": cd["sub_industry"],
                "all_industries": cd.get("all_industries", ""),
                "employee_count": cd["employees"],
                "description": cd["description"],
                "unique_roles": unique_roles,
                "fit_score": 0.0,  # Computed later by caller
                "lead_count": len(leads),
                "leads": leads,
                # Funding-specific
                "_funding_type": cd["deal_type"],
                "_funding_date": cd["deal_date"],
                "_funding_amount": cd["deal_amount"],
                "_funding_investors": investors_str,
                "_total_raised": cd["total_raised"],
                "_days_since_funding": cd["days_since"],
                "_funding_rounds": cd["num_rounds"],
            }

            companies_result.append(company_entry)
            companies_kept += 1
            logger.info(f"[Fundable] Kept {company_name} ({companies_kept}/{max_companies}, {len(leads)} leads)")

        page += 1
        if page * 100 >= (api_total or 0):
            logger.info(f"[Fundable] Reached end of API results ({api_total} total)")
            break

    logger.info(f"[Fundable] Done: {companies_kept} companies, checked {companies_checked}")
    return companies_result
