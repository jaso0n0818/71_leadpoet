"""
Configuration for the Fulfillment Lead Model.

All values are read from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# ═══════════════════════════════════════════════════════════════════════════
# OpenRouter / LLM Configuration
# ═══════════════════════════════════════════════════════════════════════════

OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "") or os.environ.get("FULFILLMENT_OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_MODEL = "google/gemini-2.5-flash-lite"
ICP_PARSER_MODEL = "anthropic/claude-sonnet-4-6"
PERPLEXITY_MODEL = "perplexity/sonar-pro"
LLM_TIMEOUT = 60
PERPLEXITY_TIMEOUT = 60
LLM_MAX_RETRIES = 3
MAX_CONCURRENT_PERPLEXITY = 3
API_DELAY = 0.5

# ═══════════════════════════════════════════════════════════════════════════
# Discovery + Intent Scoring
# ═══════════════════════════════════════════════════════════════════════════

WEIGHT_FIT_SCORE = 0.50
WEIGHT_INTENT_SCORE = 0.50
COMPANY_MULTIPLIER = 200
INTENT_THRESHOLD = 0.3

# ═══════════════════════════════════════════════════════════════════════════
# Email Verification (Truelist)
# ═══════════════════════════════════════════════════════════════════════════

TRUELIST_API_KEY = os.environ.get("TRUELIST_API_KEY", "")

# ═══════════════════════════════════════════════════════════════════════════
# Fundable API (optional)
# ═══════════════════════════════════════════════════════════════════════════

TRYFUNDABLE_API_KEY = os.environ.get("TRYFUNDABLE_API_KEY", "")

# ═══════════════════════════════════════════════════════════════════════════
# ScrapingDog
# ═══════════════════════════════════════════════════════════════════════════

SCRAPINGDOG_API_KEY = os.environ.get("SCRAPINGDOG_API_KEY", "")

# ═══════════════════════════════════════════════════════════════════════════
# Scoring Weights (must sum to 1.0)
# ═══════════════════════════════════════════════════════════════════════════

WEIGHT_INDUSTRY = 0.25
WEIGHT_SUB_INDUSTRY = 0.15
WEIGHT_ROLE = 0.25
WEIGHT_LOCATION = 0.15
WEIGHT_COMPANY_SIZE = 0.10
WEIGHT_QUALITY = 0.10

MIN_REP_SCORE = 5.0
MAX_REP_SCORE = 50.0
FETCH_MULTIPLIER = 10

# ═══════════════════════════════════════════════════════════════════════════
# Employee Count Ranges (ordered by size for distance-based matching)
# ═══════════════════════════════════════════════════════════════════════════

EMPLOYEE_RANGE_ORDER = [
    "<10",
    "10-50",
    "50-200",
    "200-500",
    "500-1000",
    "1000-5000",
    "5000-10000",
    "10000+",
]

EMPLOYEE_RANGE_ALIASES = {
    "<10 employees": "<10",
    "1-10": "<10",
    "1-10 employees": "<10",
    "10-50 employees": "10-50",
    "11-50": "10-50",
    "11-50 employees": "10-50",
    "50-200 employees": "50-200",
    "51-200": "50-200",
    "51-200 employees": "50-200",
    "200-500 employees": "200-500",
    "201-500": "200-500",
    "201-500 employees": "200-500",
    "500-1000 employees": "500-1000",
    "500-1,000 employees": "500-1000",
    "501-1000": "500-1000",
    "501-1,000": "500-1000",
    "501-1,000 employees": "500-1000",
    "1000-5000 employees": "1000-5000",
    "1,000-5,000 employees": "1000-5000",
    "1,001-5,000": "1000-5000",
    "1,001-5,000 employees": "1000-5000",
    "1001-5000": "1000-5000",
    "5000-10000 employees": "5000-10000",
    "5,000-10,000 employees": "5000-10000",
    "5,001-10,000": "5000-10000",
    "5,001-10,000 employees": "5000-10000",
    "5001-10000": "5000-10000",
    "10000+ employees": "10000+",
    "10,000+ employees": "10000+",
    "10,001+": "10000+",
    "10,001+ employees": "10000+",
    "10001+": "10000+",
}

# ═══════════════════════════════════════════════════════════════════════════
# Role Matching
# ═══════════════════════════════════════════════════════════════════════════

SENIORITY_RANK = {
    "C-Suite": 5,
    "VP": 4,
    "Director": 3,
    "Manager": 2,
    "Individual Contributor": 1,
}

ROLE_KEYWORD_EXPANSIONS = {
    "cto": {"technology", "technical", "engineering", "it"},
    "cio": {"technology", "it", "information"},
    "cdo": {"data", "analytics", "bi"},
    "cfo": {"finance", "financial"},
    "coo": {"operations", "operating"},
    "cmo": {"marketing", "growth"},
    "cro": {"revenue", "sales"},
    "cpo": {"product"},
}

ROLE_STOPWORDS = frozenset({
    "chief", "officer", "vice", "president", "vp", "svp", "evp", "head",
    "director", "managing", "manager", "senior", "lead", "principal",
    "executive", "assistant", "associate", "staff", "global", "of", "and", "the",
})

SENIORITY_KEYWORDS = {
    "C-Suite": {"chief", "ceo", "cto", "cfo", "coo", "cmo", "cro", "cpo", "cio", "cdo", "founder", "owner", "partner"},
    "VP": {"vice president", "vp", "svp", "evp"},
    "Director": {"director"},
    "Manager": {"manager", "supervisor", "team lead", "lead"},
}

# ═══════════════════════════════════════════════════════════════════════════
# Country Normalization
# ═══════════════════════════════════════════════════════════════════════════

COUNTRY_ALIASES = {
    "usa": "united states",
    "us": "united states",
    "u.s.": "united states",
    "u.s.a.": "united states",
    "united states of america": "united states",
    "uae": "united arab emirates",
}
