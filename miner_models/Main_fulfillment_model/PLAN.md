# Fulfillment Lead Model — Implementation Plan

## Overview

A model that discovers companies matching an ICP from the web, finds decision-maker contacts,
verifies emails via TrueList, and enriches with intent signals using Perplexity + ScrapingDog.
The original version queried a database; this version uses web-based discovery exclusively.

## Inputs (from UI)

| Field | Required | Column in DB | Type |
|-------|----------|-------------|------|
| Industry | **Yes** | `industry` | Exact match |
| Sub-Industry | No | `sub_industry` | Exact match |
| Role | No | `role` | Fuzzy match (keyword + seniority) |
| Country | No | `country` | Exact match |
| State | No | `state` | Exact match |
| City | No | `city` | Exact match |
| Employee Count | No | `employee_count` | Range match |
| Max Leads | **Yes** | N/A | Limit (default 5) |

## Output (per lead)

```json
{
  "first_name": "Natalie",
  "last_name": "Meyerson",
  "email": "natalie.meyerson@fanduel.com",
  "role": "Senior Financial Analyst",
  "company_name": "FanDuel",
  "linkedin": "http://www.linkedin.com/in/natalie-meyerson",
  "website": "http://www.fanduel.com",
  "company_linkedin": "https://linkedin.com/company/fanduel",
  "industry": "Gaming",
  "sub_industry": "Fantasy Sports",
  "city": "New York City",
  "state": "New York",
  "country": "United States",
  "hq_city": "New York City",
  "hq_state": "New York",
  "hq_country": "United States",
  "employee_count": "1,001-5,000 employees",
  "description": "FanDuel operates in the online sports entertainment...",
  "rep_score": 13.0,
  "fit_score": 0.92,
  "fit_breakdown": {
    "industry_match": 1.0,
    "sub_industry_match": 1.0,
    "role_match": 0.85,
    "location_match": 1.0,
    "size_match": 0.8,
    "quality_bonus": 0.13
  }
}
```

## Architecture

```
target_fit_model/
├── PLAN.md              # This file
├── __init__.py          # Package init
├── config.py            # Constants, scoring weights, employee count ranges
├── query.py             # Supabase query builder (filtering, pagination)
├── scoring.py           # Fit scoring engine (role matching, size matching)
├── model.py             # Main entry point — orchestrates query + score + rank
└── api.py               # FastAPI endpoint (optional, for direct API access)
```

## Implementation Steps

### Step 1: Config (`config.py`)

Define:
- Supabase connection (read-only, uses extracted columns NOT lead_blob)
- Employee count ranges and their ordering for range matching
- Scoring weights for each dimension
- Industry/sub-industry taxonomy (reuse from gateway)
- Seniority levels for role matching (reuse from champion model)

Key constants from champion model to reuse:
- `SENIORITY_RANK` — C-Suite=5, VP=4, Director=3, Manager=2, IC=1
- `ROLE_KEYWORD_EXPANSIONS` — CTO→{technology,engineering}, CFO→{finance}, etc.
- `ROLE_STOPWORDS` — common title words to strip for matching
- `US_STATE_NAMES` — for validation
- `_COUNTRY_ALIASES` — USA→United States, etc.

### Step 2: Query Builder (`query.py`)

Build Supabase queries using the NEW extracted columns (not lead_blob):

```python
async def query_leads(
    industry: str,
    sub_industry: Optional[str] = None,
    role: Optional[str] = None,
    country: Optional[str] = None,
    state: Optional[str] = None,
    city: Optional[str] = None,
    employee_count: Optional[str] = None,
    max_leads: int = 5,
    fetch_multiplier: int = 10,  # Fetch 10x to allow scoring/ranking
) -> List[Dict]:
```

Query strategy:
1. **Hard filters** (must match): `status = 'approved'`, `industry` (exact)
2. **Soft filters** (prefer but don't exclude): sub_industry, country, state, city
3. **Fetch `max_leads * fetch_multiplier`** candidates to allow scoring to rank them
4. **Order by `rep_score DESC`** as a tiebreaker (higher quality leads first)
5. **Use the indexed columns** for fast querying (industry, country, state, etc.)

For employee_count range matching:
- User selects "200-500 employees"
- DB has values like "201-500", "201-500 employees", "501-1,000 employees"
- Need a mapping to normalize and compare ranges

### Step 3: Fit Scoring Engine (`scoring.py`)

Score each candidate lead against the ICP. Each dimension gets a 0.0-1.0 score:

| Dimension | Weight | Scoring Logic |
|-----------|--------|---------------|
| **Industry** | 0.25 | 1.0 if exact match (always matches since it's a hard filter) |
| **Sub-Industry** | 0.15 | 1.0 exact, 0.5 if same parent industry but different sub, 0.0 if not specified |
| **Role** | 0.25 | Keyword overlap + seniority match (reuse champion's role scoring) |
| **Location** | 0.15 | 1.0 if all specified fields match, partial credit for country-only match |
| **Company Size** | 0.10 | 1.0 if exact range, 0.5 if adjacent range, 0.0 if far |
| **Quality Bonus** | 0.10 | Normalized rep_score (0-50 → 0.0-1.0) |

Role matching (from champion model):
```python
def score_role_match(lead_role: str, target_role: str) -> float:
    # 1. Extract keywords from both (strip stopwords)
    # 2. Check keyword overlap (using ROLE_KEYWORD_EXPANSIONS)
    # 3. Check seniority match (C-Suite targeting C-Suite = bonus)
    # 4. Combine: keyword_overlap * 0.6 + seniority_match * 0.4
```

Employee count range matching:
```python
EMPLOYEE_RANGES = [
    ("<10", 0), ("10-50", 1), ("50-200", 2), ("200-500", 3),
    ("500-1000", 4), ("1000-5000", 5), ("5000-10000", 6), ("10000+", 7)
]
# Distance-based: exact=1.0, adjacent=0.7, 2-away=0.4, 3+=0.0
```

### Step 4: Main Model (`model.py`)

The orchestrator:

```python
async def find_target_fit_leads(
    industry: str,
    sub_industry: Optional[str] = None,
    role: Optional[str] = None,
    country: Optional[str] = None,
    state: Optional[str] = None,
    city: Optional[str] = None,
    employee_count: Optional[str] = None,
    max_leads: int = 5,
) -> List[Dict]:
    # 1. Query candidates (fetch 10x max_leads)
    candidates = await query_leads(industry, sub_industry, ...)
    
    # 2. Score each candidate
    scored = []
    for lead in candidates:
        fit_score, breakdown = compute_fit_score(lead, {
            "industry": industry,
            "sub_industry": sub_industry,
            "role": role,
            "country": country,
            "state": state,
            "city": city,
            "employee_count": employee_count,
        })
        scored.append({**lead, "fit_score": fit_score, "fit_breakdown": breakdown})
    
    # 3. Rank by fit_score DESC, deduplicate by company
    scored.sort(key=lambda x: x["fit_score"], reverse=True)
    seen_companies = set()
    results = []
    for lead in scored:
        company = lead.get("company_name", "").lower()
        if company in seen_companies:
            continue
        seen_companies.add(company)
        results.append(lead)
        if len(results) >= max_leads:
            break
    
    return results
```

### Step 5: API Endpoint (`api.py`)

FastAPI router that integrates with the existing gateway:

```python
@router.post("/leads/target-fit")
async def get_target_fit_leads(request: TargetFitRequest):
    # Validate inputs
    # Call model
    # Return results
```

## Key Design Decisions

1. **Read-only**: NEVER writes to leads_private. Uses service role key for read access only.

2. **Uses extracted columns**: Queries `industry`, `sub_industry`, `role`, `city`, `state`, `country`, `employee_count` columns directly — NOT `lead_blob`. This leverages the JSONB migration we just completed.

3. **Company deduplication**: Only returns one lead per company to give the buyer diverse results.

4. **Quality floor**: Only returns leads with `rep_score >= 5` to ensure minimum quality.

5. **Recency preference**: Among equal-scoring leads, prefers newer ones (`created_ts DESC`).

6. **No LLM calls**: Pure database query + algorithmic scoring. Fast, deterministic, no API costs.

## What We Reuse from Champion Model

| Component | Champion Code | Our Usage |
|-----------|--------------|-----------|
| `SENIORITY_RANK` | L89 | Role seniority scoring |
| `ROLE_KEYWORD_EXPANSIONS` | L91-100 | Role keyword matching |
| `ROLE_STOPWORDS` | L102-108 | Role keyword extraction |
| `_COUNTRY_ALIASES` | L130-135 | Country normalization |
| `US_STATE_NAMES` | L110-127 | State validation |
| `_normalize_country()` | L138 | Country matching |
| Role scoring logic | L488-528 | Adapted for fit scoring |

## Estimated Timeline

- Step 1 (config): 30 min
- Step 2 (query): 1 hour
- Step 3 (scoring): 1.5 hours
- Step 4 (model): 30 min
- Step 5 (api): 30 min
- Testing: 1 hour

**Total: ~5 hours**
