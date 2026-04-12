"""
Lightweight ScrapingDog client for Google Jobs and Google News search.

Used to gather structured hiring and news data before Perplexity synthesis.
Falls back gracefully if API key not set or calls fail.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

SCRAPINGDOG_API_KEY = os.environ.get("SCRAPINGDOG_API_KEY", "").strip()
SCRAPINGDOG_BASE_URL = "https://api.scrapingdog.com"
SCRAPINGDOG_TIMEOUT = 30
API_DELAY = 0.5


def search_jobs(company_name: str) -> List[dict]:
    """
    Search Google Jobs for a company's open positions.
    Returns list of job dicts. Empty list on failure.
    Cost: 5 credits per call.
    """
    if not SCRAPINGDOG_API_KEY:
        logger.debug("[ScrapingDog] No API key, skipping jobs search")
        return []

    query = f"{company_name} jobs"
    start = time.time()

    try:
        resp = requests.get(
            f"{SCRAPINGDOG_BASE_URL}/google_jobs",
            params={"api_key": SCRAPINGDOG_API_KEY, "query": query},
            timeout=SCRAPINGDOG_TIMEOUT,
        )
        duration = time.time() - start

        if resp.status_code != 200:
            logger.warning(f"[ScrapingDog] Jobs search error {resp.status_code} for '{company_name}'")
            return []

        data = resp.json()
        results = data.get("jobs_results", data.get("results", []))
        if isinstance(data, list):
            results = data

        from target_fit_model.openrouter import track_scrapingdog
        track_scrapingdog(5)  # 5 credits per Google Jobs call
        logger.info(f"[ScrapingDog] Jobs: {len(results)} results for '{company_name}' | {duration:.1f}s")
        time.sleep(API_DELAY)
        return results

    except Exception as e:
        logger.error(f"[ScrapingDog] Jobs search error for '{company_name}': {e}")
        return []


def search_news(company_name: str) -> List[dict]:
    """
    Search Google News for recent articles about a company.
    Returns list of news article dicts. Empty list on failure.
    Cost: 5 credits per call.
    """
    if not SCRAPINGDOG_API_KEY:
        logger.debug("[ScrapingDog] No API key, skipping news search")
        return []

    start = time.time()

    try:
        resp = requests.get(
            f"{SCRAPINGDOG_BASE_URL}/google_news",
            params={"api_key": SCRAPINGDOG_API_KEY, "query": company_name},
            timeout=SCRAPINGDOG_TIMEOUT,
        )
        duration = time.time() - start

        if resp.status_code != 200:
            logger.warning(f"[ScrapingDog] News search error {resp.status_code} for '{company_name}'")
            return []

        data = resp.json()
        results = data.get("news_results", data.get("results", []))
        if isinstance(data, list):
            results = data

        from target_fit_model.openrouter import track_scrapingdog
        track_scrapingdog(5)  # 5 credits per Google News call
        logger.info(f"[ScrapingDog] News: {len(results)} results for '{company_name}' | {duration:.1f}s")
        time.sleep(API_DELAY)
        return results

    except Exception as e:
        logger.error(f"[ScrapingDog] News search error for '{company_name}': {e}")
        return []


def format_jobs_for_prompt(jobs: List[dict], company_name: str) -> str:
    """Format job results into text for Perplexity prompt."""
    if not jobs:
        return "No job postings found via Google Jobs search."

    # Basic company name token matching to filter irrelevant results
    name_tokens = set(company_name.lower().split())
    noise = {"inc", "llc", "ltd", "corp", "the", "group", "co", "company"}
    name_tokens -= noise

    relevant = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        employer = (job.get("company_name") or job.get("employer", "")).lower()
        if any(t in employer for t in name_tokens if len(t) > 2):
            title = job.get("title", job.get("job_title", "Unknown"))
            location = job.get("location", "")
            link = job.get("link", job.get("job_link", ""))
            relevant.append(f"- {title} ({location}) {link}")

    if not relevant:
        return f"Google Jobs returned results but none confirmed for {company_name}."

    header = f"{len(relevant)} job postings found for {company_name}:"
    return header + "\n" + "\n".join(relevant[:15])


def format_news_for_prompt(news: List[dict], company_name: str) -> str:
    """Format news results into text for Perplexity prompt."""
    if not news:
        return "No recent news articles found via Google News search."

    name_tokens = set(company_name.lower().split())
    noise = {"inc", "llc", "ltd", "corp", "the", "group", "co", "company"}
    name_tokens -= noise

    relevant = []
    for article in news:
        if not isinstance(article, dict):
            continue
        title = (article.get("title") or "").lower()
        snippet = (article.get("snippet") or article.get("description", "")).lower()
        text = title + " " + snippet

        if any(t in text for t in name_tokens if len(t) > 2):
            a_title = article.get("title", "Unknown")
            a_source = article.get("source", article.get("publisher", ""))
            a_date = article.get("date", article.get("published_date", ""))
            a_link = article.get("link", article.get("url", ""))
            relevant.append(f"- {a_title} ({a_source}, {a_date}) {a_link}")

    if not relevant:
        return f"Google News returned results but none confirmed for {company_name}."

    header = f"{len(relevant)} news articles found for {company_name}:"
    return header + "\n" + "\n".join(relevant[:10])
