"""
Intent enrichment: Multi-source research + Perplexity synthesis.

Flow per company:
1. ScrapingDog Google Jobs → hiring signals (structured)
2. ScrapingDog Google News → news signals (structured)
3. Perplexity sonar-pro → synthesize all data + search social media + score signals

Perplexity receives raw jobs/news data as context, validates it,
searches additional sources (LinkedIn, Twitter, blogs), and outputs
a single polished paragraph with evidence and source URLs.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from target_fit_model.config import (
    PERPLEXITY_MODEL,
    PERPLEXITY_DEEP_MODEL,
    PERPLEXITY_TIMEOUT,
    PERPLEXITY_DEEP_TIMEOUT,
    MAX_CONCURRENT_PERPLEXITY,
)
from target_fit_model.openrouter import chat_completion, chat_completion_json, parse_json_response
from target_fit_model.scrapingdog import (
    search_jobs,
    search_news,
    format_jobs_for_prompt,
    format_news_for_prompt,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a business research analyst helping a sales team identify "
    "companies with active buying signals. Research thoroughly using all "
    "available web sources. Every claim MUST have a source URL. "
    "Return ONLY valid JSON."
)


def research_company_intent(
    company: Dict,
    product_description: str,
    request_description: str,
    intent_signals: str,
) -> Optional[Dict]:
    """
    Research a single company's intent signals using multi-source approach.

    1. Fetch structured data (Google Jobs, Google News) via ScrapingDog
    2. Send all data to Perplexity for synthesis + additional research
    3. Returns scores, intent paragraph with evidence + URLs
    """
    company_name = company.get("company_name") or company.get("company") or "Unknown"
    website = company.get("website") or ""
    linkedin = company.get("company_linkedin") or ""
    industry = company.get("industry") or ""
    description = company.get("description") or ""

    # Dynamic recency cutoff
    from datetime import datetime, timedelta
    _six_months_ago = (datetime.now() - timedelta(days=180)).strftime("%B %Y")

    # ── Step 1: Gather structured data ──
    logger.info(f"[Intent] Researching {company_name}...")

    jobs_data = search_jobs(company_name)
    news_data = search_news(company_name)

    jobs_text = format_jobs_for_prompt(jobs_data, company_name)
    news_text = format_news_for_prompt(news_data, company_name)

    # ── Step 2: Perplexity synthesis ──
    prompt = f"""## Product Being Sold
{product_description}

## Target Company Profile
{request_description}

## Company to Research
Name: "{company_name}"
Website: {website}
LinkedIn: {linkedin}
Industry: {industry}
Description: {description}

## Data Already Gathered

### Job Postings (from Google Jobs):
{jobs_text}

### Recent News (from Google News):
{news_text}

### Funding Data (pre-verified):
{company.get("_funding_evidence") or "No funding data gathered"}

## Intent Signals to Evaluate
{intent_signals}

## Instructions

Conduct a THOROUGH deep research investigation of this company. Go beyond surface-level searches.

IMPORTANT:
1. VALIDATE all job postings and news articles — confirm they are actually about "{company_name}", not a different company with a similar name. Discard any that don't match.
2. Visit and analyze these sources IN DEPTH:
   - Company website (about page, careers page, blog, news section)
   - LinkedIn company page — recent posts, employee growth, job postings
   - X/Twitter company account — recent tweets, engagement
   - Glassdoor reviews — culture signals, restructuring, leadership changes
   - Crunchbase/PitchBook — funding rounds, investors, valuation
   - SEC filings / earnings reports (if public)
   - Google News — press releases, media coverage, partnerships
   - YouTube channel — webinars, product demos, thought leadership
   - Industry forums, Reddit, community discussions
   - Job boards (Indeed, LinkedIn Jobs, Glassdoor Jobs)
3. SOCIAL MEDIA RESEARCH — follow these steps:
   a. First, identify the CEO/founder of "{company_name}" by searching their website's about page or team page
   b. Search for that person's name + "linkedin.com" to find their LinkedIn profile
   c. Search for "{company_name}" + CEO name + recent posts, interviews, podcasts, or quotes about topics related to the intent signals
   d. Search for "{company_name}" on X/Twitter for recent company announcements
   e. If you find any executive publicly discussing topics related to the product being sold, QUOTE THEM DIRECTLY with the source
   f. Only include quotes where the executive discusses topics DIRECTLY related to needing the product — not general business commentary
4. EVERY claim must include a source URL. No unverifiable statements.
5. RECENCY IS CRITICAL: Only include evidence from the last 6 months (since {_six_months_ago}). Include the date (month/year) with each claim. Ignore anything older than 6 months.
6. RELEVANCE IS CRITICAL: Only include signals that directly suggest this company would BENEFIT FROM or NEED the product being sold. Do NOT include general business news, competitive positioning, or executive statements that don't indicate a specific need for the product. Ask yourself: "Would a sales person use this evidence to pitch the product?" — if not, exclude it.
7. Think about INDIRECT signals but only if they logically connect to needing the product: team restructuring (need training for new team), rapid hiring (need onboarding methodology), leadership changes (new leader brings new processes), product launch (need new sales motion).

Return this exact JSON:

{{
  "signals": [
    {{
      "signal": "the intent signal being evaluated",
      "match": true or false,
      "relevance_score": 0.0 to 1.0,
      "evidence": "brief factual evidence with date (month/year) and source URL inline, or null if none. Only evidence from last 6 months."
    }}
  ],
  "intent_paragraph": "A single polished, client-ready paragraph focused ONLY on positive buying signals found. Explain WHY each signal indicates relevance for the product being sold. Weave source URLs naturally into the text after each claim. Do NOT list signals that were not found. Do NOT say 'no evidence found', 'no active buying signals', 'unavailable', or 'at this time'. Only include what WAS found. If nothing was found, write a brief neutral description of the company's current market position and recent activities. No bullets, no labels, no em dashes, no subtitles, no hyphens overuse. Do not reference 'this list' or 'the source file'. Do not restate ICP facts like country or employee count. Ensure all details are factual and evidence-based. Write as if this will appear directly in a client-facing sales intelligence report.",
  "intent_score": 0.0 to 1.0,
  "fit_score": 0.0 to 1.0,
  "overall_confidence": 0.0 to 1.0
}}

Scoring guide for relevance_score per signal:
- 1.0 = strong, recent, confirmed evidence (job posting live, press release, public announcement)
- 0.7 = partial or somewhat dated evidence
- 0.3 = weak/indirect signal (industry trend, tangential mention)

IMPORTANT: Only include signals where you found evidence (score >= 0.3). OMIT signals with no evidence — do not include them with score 0.0. This keeps the response focused on what was actually found.

intent_score: average of matched signal relevance_scores (only signals with evidence)
fit_score: how well this company matches the target profile and would benefit from the product (0-1)"""

    # Track data gaps
    data_gaps = []
    if not jobs_data:
        data_gaps.append("No job postings found via Google Jobs")
    if not news_data:
        data_gaps.append("No recent news found via Google News")

    def _parse_result(result, data_gaps):
        """Helper to validate and normalize Perplexity result."""
        # Handle string responses — extract intent_paragraph directly
        if result and isinstance(result, str) and len(result) > 50:
            return {
                "signals": [],
                "intent_paragraph": result.strip(),
                "intent_score": 0.5,
                "fit_score": 0.5,
                "_data_gaps": data_gaps,
            }

        if result and isinstance(result, dict) and "signals" in result:
            if "intent_paragraph" not in result:
                result["intent_paragraph"] = ""
            if "intent_score" not in result:
                result["intent_score"] = compute_intent_score_from_signals(result.get("signals", []))
            if "fit_score" not in result:
                result["fit_score"] = 0.5
            result["_data_gaps"] = data_gaps
            return result

        if result and isinstance(result, dict):
            result.setdefault("signals", [])
            result.setdefault("intent_paragraph", "")
            if "intent_score" not in result:
                result["intent_score"] = compute_intent_score_from_signals(result.get("signals", []))
            result.setdefault("fit_score", 0.5)
            result["_data_gaps"] = data_gaps
            return result

        return None

    # Attempt 1: Deep research to gather real evidence, then extract JSON
    print(f"    [Intent] Attempt 1 for {company_name} (jobs={bool(jobs_data)}, news={bool(news_data)})")

    deep_research_prompt = f"""Find VERIFIABLE evidence that "{company_name}" is showing these buying signals: {intent_signals}

Context: A sales team selling "{product_description}" wants proof these signals are real.

CRITICAL URL RULES:
1. Each signal MUST link to a DIFFERENT domain — the validator rejects multiple signals from the same domain
2. STRONGLY PREFER third-party sources: news (TechCrunch, Forbes, Bloomberg, CRN, VentureBeat), job boards (Indeed, Greenhouse, Lever, LinkedIn Jobs), press releases (PRNewsWire, BusinessWire), Crunchbase, G2, TrustRadius
3. The company's own website ({website}) is allowed ONLY for specific deep pages (e.g. /careers/sdr-role, /blog/2026-expansion) — NEVER use the homepage or generic pages like /about, /press, /careers
4. Each URL must be a specific page that mentions "{company_name}" by name with verifiable facts
5. Only include evidence from the last 6 months (since {_six_months_ago})

Source strategy per signal type:
- Hiring signals → Indeed.com, Greenhouse.io, Lever.co, or LinkedIn Jobs (actual job postings)
- Funding signals → Crunchbase.com, TechCrunch.com, PRNewsWire.com (announcements)
- Competitor/tool evaluation → G2.com, TrustRadius.com, news articles
- General business signals → News articles, LinkedIn company posts, press releases

Return ONLY this JSON:
{{"signals": [{{"signal": "signal name", "match": true, "relevance_score": 0.7, "evidence": "what you found, with the date (month/year) and full source URL"}}], "intent_paragraph": "one paragraph summary", "intent_score": 0.5, "fit_score": 0.5}}

If no verifiable evidence found, return: {{"signals": [], "intent_paragraph": "", "intent_score": 0.0, "fit_score": 0.0}}"""

    # Try deep research first
    from target_fit_model.openrouter import chat_completion
    deep_raw = chat_completion(
        prompt=deep_research_prompt,
        model=PERPLEXITY_DEEP_MODEL,
        system_prompt="You are a sales research assistant. Return ONLY valid JSON.",
        temperature=0,
        max_tokens=8000,
        timeout=PERPLEXITY_DEEP_TIMEOUT,
    )

    result = None
    if deep_raw and len(deep_raw) > 20:
        from target_fit_model.openrouter import parse_json_response
        result = parse_json_response(deep_raw)

        if result is None and len(deep_raw) > 200:
            # Deep research returned markdown — extract URLs and evidence,
            # then ask a fast model to structure it as JSON.
            import re as _re
            urls_found = _re.findall(r'https?://[^\s)<>\]"]+', deep_raw)
            urls_snippet = "\n".join(f"- {u}" for u in urls_found[:15])
            company_domain = (website.replace("https://", "").replace("http://", "")
                              .split("/")[0].lower() if website else "")

            # Separate third-party URLs from company URLs
            third_party_urls = [u for u in urls_found if company_domain not in u.lower()]
            company_urls = [u for u in urls_found if company_domain in u.lower()
                           and u.lower().rstrip("/") != f"https://{company_domain}"
                           and u.lower().rstrip("/") != f"http://{company_domain}"]
            urls_snippet = "\n".join(f"- {u}" for u in (third_party_urls + company_urls)[:15])

            extract_prompt = f"""Extract intent signals from this research about "{company_name}".

RESEARCH TEXT:
{deep_raw[:6000]}

URLs found in the research:
{urls_snippet}

Signals to look for: {intent_signals}

RULES:
- Each signal MUST use a URL from a DIFFERENT domain (validator skips duplicate domains)
- PREFER third-party URLs (news, job boards, Crunchbase, etc.) over company website URLs
- Company website URLs ({company_domain}) are OK only for specific deep pages (e.g. /careers/specific-job, /blog/specific-post) — NOT homepages or generic pages
- The evidence field must include the full source URL
- Only include signals with real, specific evidence

Return ONLY this JSON:
{{"signals": [{{"signal": "signal name", "match": true, "relevance_score": 0.7, "evidence": "specific evidence with date and source URL"}}], "intent_paragraph": "brief summary of findings", "intent_score": 0.5, "fit_score": 0.5}}

If no verifiable evidence found, return: {{"signals": [], "intent_paragraph": "", "intent_score": 0.0, "fit_score": 0.0}}"""

            result = chat_completion_json(
                prompt=extract_prompt,
                model=PERPLEXITY_MODEL,
                system_prompt="Extract structured data from research text. Return ONLY valid JSON.",
                temperature=0,
                max_tokens=4000,
                timeout=PERPLEXITY_TIMEOUT,
            )
            if result:
                print(f"    [Intent] Extracted JSON from deep research markdown")

    parsed = _parse_result(result, data_gaps)
    if parsed:
        sig_count = len(parsed.get("signals", []))
        matched = sum(1 for s in parsed.get("signals", []) if s.get("match"))
        print(f"    [Intent] Attempt 1 result: {sig_count} signals ({matched} matched)")
        return parsed

    # Attempt 2: Retry with shorter prompt (original signals only, no ScrapingDog data)
    print(f"    [Intent] Attempt 1 failed (result={type(result).__name__}: {str(result)[:100] if result else 'None'}), retrying...")

    short_prompt = f"""## Product Being Sold
{product_description}

## Company to Research
Name: "{company_name}"
Website: {website}
LinkedIn: {linkedin}
Industry: {industry}

## Intent Signals to Evaluate
{intent_signals}

Research this company for the intent signals above. Only include signals where you find evidence from the last 6 months. Include date and source URL for each claim.

Return JSON:
{{
  "signals": [
    {{
      "signal": "signal description",
      "match": true,
      "relevance_score": 0.3 to 1.0,
      "evidence": "evidence with date and source URL"
    }}
  ],
  "intent_paragraph": "Single polished paragraph with source URLs inline. Only positive findings. Client-ready.",
  "intent_score": 0.0 to 1.0,
  "fit_score": 0.0 to 1.0,
  "overall_confidence": 0.0 to 1.0
}}"""

    result = chat_completion_json(
        prompt=short_prompt,
        model=PERPLEXITY_MODEL,
        system_prompt=_SYSTEM_PROMPT,
        temperature=0,
        max_tokens=4000,
        timeout=PERPLEXITY_TIMEOUT,
    )

    parsed = _parse_result(result, data_gaps)
    if parsed:
        sig_count = len(parsed.get("signals", []))
        matched = sum(1 for s in parsed.get("signals", []) if s.get("match"))
        print(f"    [Intent] Attempt 2 result: {sig_count} signals ({matched} matched)")
        return parsed

    # Both attempts failed — return empty dict (NEVER return string or None)
    print(f"    [Intent] Both attempts FAILED for {company_name} (result={type(result).__name__}: {str(result)[:100] if result else 'None'})")
    return {
        "signals": [],
        "intent_paragraph": "",
        "intent_score": 0.0,
        "fit_score": 0.5,
        "overall_confidence": 0.0,
        "_data_gaps": data_gaps + ["Perplexity research failed after 2 attempts"],
    }


def compute_intent_score_from_signals(signals: List[Dict], original_signals: Optional[List[str]] = None) -> float:
    """
    Compute intent_score using tiered weighted approach:
    - Tier 1 (user's original signals): weight 2x
    - Tier 2 (Gemini expanded signals): weight 1x
    - Only counts matched signals (score > 0)
    - Coverage bonus (30%) rewards breadth of evidence

    Args:
        signals: List of signal dicts from Perplexity
        original_signals: List of user's original signal strings (for tier detection)
    """
    if not signals:
        return 0.0

    # Build lowercase set of original signals for matching
    orig_lower = set()
    if original_signals:
        for s in original_signals:
            orig_lower.add(s.strip().lower())

    weighted_scores = []
    total_weight = 0.0

    for sig in signals:
        score = sig.get("relevance_score")
        if not isinstance(score, (int, float)) or float(score) <= 0:
            continue

        score = max(0.0, min(1.0, float(score)))
        signal_text = (sig.get("signal") or "").strip().lower()

        # Tier 1: user's original signal — weight 2x
        # Tier 2: expanded signal — weight 1x
        is_tier1 = False
        if orig_lower:
            for orig in orig_lower:
                if orig in signal_text or signal_text in orig:
                    is_tier1 = True
                    break

        weight = 2.0 if is_tier1 else 1.0
        weighted_scores.append(score * weight)
        total_weight += weight

    if not weighted_scores:
        return 0.0

    quality = sum(weighted_scores) / total_weight
    matched_count = len(weighted_scores)
    total_count = len(signals) if signals else 1
    coverage = matched_count / total_count

    return round(quality * 0.70 + coverage * 0.30, 4)


def _cross_source_boost(signals: List[Dict], base_score: float) -> Tuple[float, int]:
    """
    Apply cross-source boost when 2+ distinct source domains confirm intent.
    Returns (boosted_score, source_count).
    """
    source_domains = set()
    for sig in signals:
        if not sig.get("match"):
            continue
        # Check evidence and source_url for domain hints
        evidence = (sig.get("evidence") or "") + " " + (sig.get("source_url") or "")
        evidence_lower = evidence.lower()

        if "linkedin.com" in evidence_lower:
            source_domains.add("linkedin")
        if "indeed.com" in evidence_lower or "google.com/jobs" in evidence_lower:
            source_domains.add("jobs")
        if any(d in evidence_lower for d in ["news", "techcrunch", "bloomberg", "reuters", "prnewswire", "globenewswire", "businesswire"]):
            source_domains.add("news")
        if "glassdoor" in evidence_lower:
            source_domains.add("glassdoor")
        if "twitter.com" in evidence_lower or "x.com" in evidence_lower:
            source_domains.add("social")
        if "/careers" in evidence_lower or "/jobs" in evidence_lower:
            source_domains.add("careers_page")
        if "blog" in evidence_lower:
            source_domains.add("blog")
        if "youtube.com" in evidence_lower:
            source_domains.add("youtube")

    boosted = base_score
    if len(source_domains) >= 2:
        boosted = min(boosted * 1.15, 1.0)
    if len(source_domains) >= 3:
        boosted = min(boosted * 1.10, 1.0)

    return round(boosted, 4), len(source_domains)


def compute_lead_score(intent_score: float, fit_score: float) -> float:
    """Combine intent and fit scores. Intent weighs more since DB already filtered for basic fit."""
    return round(intent_score * 0.60 + fit_score * 0.40, 4)


def enrich_companies(
    companies: List[Dict],
    product_description: str,
    request_description: str,
    intent_signals: str,
) -> List[Dict]:
    """
    Enrich multiple companies with intent signals.
    Runs concurrently with MAX_CONCURRENT_PERPLEXITY workers.
    """
    def _research_one(company: Dict) -> Dict:
        company_name = company.get("company_name") or company.get("company") or "Unknown"
        try:
            result = research_company_intent(
                company=company,
                product_description=product_description,
                request_description=request_description,
                intent_signals=intent_signals,
            )

            if result:
                signals = result.get("signals", [])
                raw_intent = float(result.get("intent_score", 0.0))
                fit_score = float(result.get("fit_score", 0.5))

                # Cross-source boost
                boosted_intent, source_count = _cross_source_boost(signals, raw_intent)

                # Track data gaps
                data_gaps = result.get("_data_gaps", [])

                return {
                    **company,
                    "intent_score": boosted_intent,
                    "fit_score_perplexity": fit_score,
                    "lead_score": compute_lead_score(boosted_intent, fit_score),
                    "intent_paragraph": result.get("intent_paragraph", ""),
                    "intent_signals_detail": signals,
                    "overall_confidence": result.get("overall_confidence", 0.0),
                    "_source_count": source_count,
                    "_data_gaps": data_gaps,
                    "_intent_failed": False,
                }
            else:
                return {
                    **company,
                    "intent_score": 0.5,
                    "fit_score_perplexity": 0.5,
                    "lead_score": 0.5,
                    "intent_paragraph": "Intent research unavailable for this company.",
                    "intent_signals_detail": [],
                    "overall_confidence": 0.0,
                    "_intent_failed": True,
                }
        except Exception as e:
            logger.error(f"[Intent] Exception for {company_name}: {e}")
            return {
                **company,
                "intent_score": 0.5,
                "fit_score_perplexity": 0.5,
                "lead_score": 0.5,
                "intent_paragraph": "Intent research failed for this company.",
                "intent_signals_detail": [],
                "overall_confidence": 0.0,
                "_intent_failed": True,
            }

    enriched = [None] * len(companies)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PERPLEXITY) as executor:
        futures = {executor.submit(_research_one, c): i for i, c in enumerate(companies)}
        for future in as_completed(futures):
            idx = futures[future]
            enriched[idx] = future.result()

    enriched = [r for r in enriched if r is not None]
    failed = sum(1 for r in enriched if r.get("_intent_failed"))
    if failed:
        logger.warning(f"[Intent] {failed}/{len(companies)} companies failed")

    # Sort by lead_score descending
    enriched.sort(key=lambda x: x.get("lead_score", 0), reverse=True)

    return enriched
