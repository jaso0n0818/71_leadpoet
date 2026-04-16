"""
TrueList email verification — lightweight version for target-fit leads.

Submits a batch of emails to TrueList API, polls for results,
and returns pass/fail per email.

No database writes — only API calls to TrueList.
"""

import asyncio
import csv
import json
import logging
import os
import time
import uuid
from datetime import datetime
from io import StringIO
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

TRUELIST_API_KEY = os.environ.get("TRUELIST_API_KEY", "")
TRUELIST_BASE_URL = "https://api.truelist.io/api/v1"
TRUELIST_POLL_INTERVAL = 10  # seconds
TRUELIST_TIMEOUT = 300  # 5 minutes max


PASS_STATUSES = {"email_ok"}
FAIL_STATUSES = {"disposable", "failed_no_mailbox", "failed_syntax_check", "failed_mx_check",
                 "accept_all", "role_based", "is_role", "catch_all"}


def verify_emails_inline(emails: List[str]) -> Dict[str, dict]:
    """
    Verify 1-3 emails using Truelist inline API.
    Works for single emails (unlike batch which needs 2+).
    """
    if not TRUELIST_API_KEY:
        logger.warning("TRUELIST_API_KEY not set")
        return {}

    if not emails:
        return {}

    batch = emails[:3]
    email_param = " ".join(batch)

    MAX_RETRIES = 2
    for attempt in range(MAX_RETRIES):
      try:
        url = f"{TRUELIST_BASE_URL}/verify_inline?email={email_param}"
        headers = {"Authorization": f"Bearer {TRUELIST_API_KEY}"}
        resp = requests.post(url, headers=headers, timeout=30)

        if resp.status_code == 429:
            logger.warning(f"[Truelist] Rate limited on inline, attempt {attempt+1}, waiting 3s")
            time.sleep(3)
            if attempt < MAX_RETRIES - 1:
                continue
            return {e.lower(): {"passed": True, "status": "rate_limited"} for e in batch}

        if resp.status_code != 200:
            logger.error(f"[Truelist] Inline error {resp.status_code}: {resp.text[:200]}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
            return {}

        data = resp.json()
        results = {}

        # Response format: {"emails": [{"address": "...", "email_state": "...", "email_sub_state": "..."}]}
        items = data.get("emails", []) if isinstance(data, dict) else data
        for item in items:
            email = (item.get("address") or item.get("email") or "").lower()
            status = (item.get("email_sub_state") or item.get("email_state") or "unknown").lower()
            passed = status in PASS_STATUSES
            if not email:
                continue
            results[email] = {
                "passed": passed,
                "status": status,
                "rejection_reason": None if passed else status,
            }

        logger.info(f"[Truelist] Inline: {len(results)} results for {len(batch)} emails")
        return results

      except Exception as e:
        logger.error(f"[Truelist] Inline error (attempt {attempt+1}): {e}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(2)
            continue
        return {}


def verify_emails_batch(emails: List[str]) -> Dict[str, dict]:
    """
    Verify a batch of emails via TrueList.

    Args:
        emails: List of email addresses

    Returns:
        Dict mapping email → {
            "passed": bool,
            "status": str,  # "email_ok", "disposable", "failed_no_mailbox", etc.
            "rejection_reason": str or None,
        }
    """
    if not TRUELIST_API_KEY:
        logger.warning("TRUELIST_API_KEY not set, skipping email verification")
        return {e: {"passed": True, "status": "skipped", "rejection_reason": None} for e in emails}

    if not emails:
        return {}

    # Filter valid emails
    valid_emails = [e.strip() for e in emails if "@" in e and e.strip()]
    invalid_emails = [e for e in emails if "@" not in e]

    results = {}

    # Mark invalid emails immediately
    for e in invalid_emails:
        results[e] = {
            "passed": False,
            "status": "invalid_syntax",
            "rejection_reason": "Email missing @ symbol",
        }

    if not valid_emails:
        return results

    # Step 1: Submit batch
    headers = {"Authorization": f"Bearer {TRUELIST_API_KEY}"}
    unique_name = f"targetfit_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    payload = {
        "data": [[e] for e in valid_emails],
        "validation_strategy": "accurate",
        "name": unique_name,
    }

    try:
        resp = requests.post(
            f"{TRUELIST_BASE_URL}/batches",
            headers=headers,
            json=payload,
            timeout=60,
        )

        if resp.status_code != 200:
            logger.error(f"TrueList submit failed: HTTP {resp.status_code} - {resp.text[:200]}")
            for e in valid_emails:
                results[e] = {"passed": False, "status": "api_error", "rejection_reason": "TrueList API error"}
            return results

        batch_id = resp.json().get("id")
        if not batch_id:
            logger.error("TrueList: no batch_id in response")
            for e in valid_emails:
                results[e] = {"passed": True, "status": "api_error", "rejection_reason": None}
            return results

        logger.info(f"TrueList batch submitted: {batch_id} ({len(valid_emails)} emails)")

    except Exception as e:
        logger.error(f"TrueList submit error: {e}")
        for e_addr in valid_emails:
            results[e_addr] = {"passed": True, "status": "api_error", "rejection_reason": None}
        return results

    # Step 2: Poll for completion
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > TRUELIST_TIMEOUT:
            logger.warning(f"TrueList polling timed out after {TRUELIST_TIMEOUT}s")
            for e in valid_emails:
                if e not in results:
                    results[e] = {"passed": False, "status": "timeout", "rejection_reason": "TrueList verification timed out"}
            return results

        time.sleep(TRUELIST_POLL_INTERVAL)

        try:
            resp = requests.get(
                f"{TRUELIST_BASE_URL}/batches/{batch_id}",
                headers=headers,
                timeout=30,
            )

            if resp.status_code != 200:
                continue

            data = resp.json()
            batch_state = data.get("batch_state", "")
            email_count = data.get("email_count", 0)
            processed_count = data.get("processed_count", 0)

            logger.info(f"TrueList poll: {batch_state} ({processed_count}/{email_count})")

            if batch_state == "completed" and processed_count >= email_count:
                # Wait for CSV generation
                time.sleep(10)

                # Re-fetch for fresh CSV URLs
                resp = requests.get(
                    f"{TRUELIST_BASE_URL}/batches/{batch_id}",
                    headers=headers,
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()

                # Download CSV
                csv_url = (
                    data.get("annotated_csv_url")
                    or data.get("results_url")
                    or data.get("download_url")
                )

                if csv_url:
                    csv_results = _download_and_parse_csv(csv_url, headers)
                    results.update(csv_results)
                else:
                    # Try combining highest_reach + invalid CSVs
                    for url_key in ["highest_reach_csv_url", "only_invalid_csv_url"]:
                        url = data.get(url_key)
                        if url:
                            partial = _download_and_parse_csv(url, headers)
                            results.update(partial)

                # Fill any missing emails as passed
                for e in valid_emails:
                    if e.lower() not in {k.lower() for k in results}:
                        results[e] = {"passed": True, "status": "not_in_results", "rejection_reason": None}

                return results

        except Exception as e:
            logger.error(f"TrueList poll error: {e}")
            continue

    return results


def _download_and_parse_csv(csv_url: str, headers: dict) -> Dict[str, dict]:
    """Download and parse a TrueList CSV response."""
    results = {}

    PASS_STATUSES = {"email_ok"}  # Only email_ok passes — matches leadpoet
    FAIL_STATUSES = {
        "disposable", "failed_no_mailbox", "failed_syntax_check",
        "failed_mx_check", "accept_all", "role_based", "is_role",
    }

    try:
        resp = requests.get(csv_url, headers=headers, timeout=60)
        if resp.status_code != 200:
            return results

        reader = csv.DictReader(StringIO(resp.text))

        # Find email and status columns
        fieldnames = reader.fieldnames or []
        email_col = None
        status_col = None

        for col in fieldnames:
            cl = col.lower().strip()
            if cl in ("email address", "email", "email_address"):
                email_col = col
            if cl in ("email sub-state", "email_sub_state", "email state", "email_state"):
                status_col = col

        if not email_col or not status_col:
            return results

        for row in reader:
            email = (row.get(email_col) or "").strip().lower()
            status = (row.get(status_col) or "").strip().lower()

            if not email:
                continue

            if status in PASS_STATUSES:
                results[email] = {"passed": True, "status": status, "rejection_reason": None}
            elif status in FAIL_STATUSES:
                results[email] = {
                    "passed": False,
                    "status": status,
                    "rejection_reason": f"Email verification failed: {status}",
                }
            else:
                # Unknown status — pass through
                results[email] = {"passed": True, "status": status, "rejection_reason": None}

    except Exception as e:
        logger.error(f"CSV parse error: {e}")

    return results
