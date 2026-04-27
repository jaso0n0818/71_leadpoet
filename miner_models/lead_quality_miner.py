from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

try:
    import httpx
except ImportError:  # Offline dry runs can still exercise local quality rules.
    httpx = None

try:
    from Leadpoet.utils.cloud_db import (
        check_email_duplicate,
        check_linkedin_combo_duplicate,
        gateway_get_presigned_url,
        gateway_upload_lead,
        gateway_verify_submission,
        get_rejection_feedback,
    )
except ImportError:
    def check_email_duplicate(email: str) -> bool:
        return False

    def check_linkedin_combo_duplicate(linkedin: str, company_linkedin: str) -> bool:
        return False

    def gateway_get_presigned_url(wallet: Any, lead_data: Dict) -> Dict:
        return {}

    def gateway_upload_lead(presigned_url: str, lead_data: Dict) -> bool:
        return False

    def gateway_verify_submission(wallet: Any, lead_id: str) -> Dict:
        return {}

    def get_rejection_feedback(wallet: Any, limit: int = 50) -> List[Dict]:
        return []

try:
    from miner_models.lead_sorcerer_main.main_leads import get_leads
except ImportError:
    async def get_leads(count: int, industry: Optional[str] = None, region: Optional[str] = None) -> List[Dict]:
        return []

logger = logging.getLogger(__name__)


GENERIC_EMAIL_PREFIXES = {
    "admin",
    "contact",
    "hello",
    "help",
    "info",
    "office",
    "sales",
    "support",
    "team",
}

DECISION_MAKER_ROLES = {
    "ceo",
    "chief executive officer",
    "founder",
    "co-founder",
    "co founder",
    "cto",
    "chief technology officer",
    "coo",
    "chief operating officer",
    "cmo",
    "chief marketing officer",
    "vp sales",
    "vp of sales",
    "vice president sales",
    "vice president of sales",
    "head of growth",
}

VALID_EMPLOYEE_COUNTS = {
    "0-1",
    "2-10",
    "11-50",
    "51-200",
    "201-500",
    "501-1,000",
    "1,001-5,000",
    "5,001-10,000",
    "10,001+",
}

EMPLOYEE_COUNT_ALIASES = {
    "1-10": "2-10",
    "10-50": "11-50",
    "50-200": "51-200",
    "200-500": "201-500",
    "500-1000": "501-1,000",
    "501-1000": "501-1,000",
    "1000-5000": "1,001-5,000",
    "1001-5000": "1,001-5,000",
    "5000-10000": "5,001-10,000",
    "5001-10000": "5,001-10,000",
    "10000+": "10,001+",
}


@dataclass
class LeadDecision:
    accepted: bool
    confidence_score: int
    reasons: List[str] = field(default_factory=list)
    lead: Optional[Dict[str, Any]] = None


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _lower(value: Any) -> str:
    return _clean(value).lower()


def _ensure_https(url: str) -> str:
    url = _clean(url)
    if not url:
        return ""
    if not re.match(r"^https?://", url, flags=re.I):
        url = f"https://{url}"
    return url


def _domain_from_url(url: str) -> str:
    try:
        host = urlparse(_ensure_https(url)).hostname or ""
        host = host.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _email_domain(email: str) -> str:
    email = _lower(email)
    return email.rsplit("@", 1)[1] if "@" in email else ""


def _registrableish_domain(domain: str) -> str:
    parts = (domain or "").lower().split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else domain.lower()


def _lead_hash(lead: Dict[str, Any]) -> str:
    stable = {
        "email": _lower(lead.get("email")),
        "linkedin": _lower(lead.get("linkedin")),
        "company_linkedin": _lower(lead.get("company_linkedin")),
        "business": _lower(lead.get("business")),
    }
    blob = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class LocalJsonSet:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.values: Set[str] = set()
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.values = set()
            return
        try:
            data = json.loads(self.path.read_text())
            if isinstance(data, dict):
                data = data.get("hashes", [])
            self.values = {_lower(v) for v in data if v}
        except Exception as exc:
            logger.warning("Failed to read %s: %s", self.path, exc)
            self.values = set()

    def add(self, value: str) -> None:
        value = _lower(value)
        if value:
            self.values.add(value)
            self.save()

    def contains(self, value: str) -> bool:
        return _lower(value) in self.values

    def save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(sorted(self.values), indent=2))
        tmp.replace(self.path)


class SuppressionList:
    def __init__(self, path: Path):
        self.path = path
        self.emails: Set[str] = set()
        self.domains: Set[str] = set()
        self.linkedin: Set[str] = set()
        self.company_linkedin: Set[str] = set()
        self.hashes: Set[str] = set()
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except Exception as exc:
            logger.warning("Failed to read suppression list %s: %s", self.path, exc)
            return

        if isinstance(data, list):
            self.emails = {_lower(v) for v in data}
            return

        self.emails = {_lower(v) for v in data.get("emails", [])}
        self.domains = {_lower(v) for v in data.get("domains", [])}
        self.linkedin = {_lower(v) for v in data.get("linkedin", [])}
        self.company_linkedin = {_lower(v) for v in data.get("company_linkedin", [])}
        self.hashes = {_lower(v) for v in data.get("hashes", [])}

    def blocks(self, lead: Dict[str, Any]) -> Optional[str]:
        email = _lower(lead.get("email"))
        website_domain = _domain_from_url(lead.get("website", ""))
        person_li = _lower(lead.get("linkedin"))
        company_li = _lower(lead.get("company_linkedin"))
        h = _lead_hash(lead)

        if email and email in self.emails:
            return "suppressed_email"
        if website_domain and website_domain in self.domains:
            return "suppressed_domain"
        if person_li and person_li in self.linkedin:
            return "suppressed_person_linkedin"
        if company_li and company_li in self.company_linkedin:
            return "suppressed_company_linkedin"
        if h in self.hashes:
            return "suppressed_hash"
        return None


class RejectionFeedbackLearner:
    def __init__(self, path: Path, min_pattern_count: int = 2):
        self.path = path
        self.min_pattern_count = min_pattern_count
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state: Dict[str, Any] = {
            "processed_ids": [],
            "reason_counts": {},
            "domain_counts": {},
            "role_counts": {},
            "email_local_counts": {},
            "last_sync_at": None,
            "examples": [],
        }
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            if isinstance(data, dict):
                self.state.update(data)
        except Exception as exc:
            logger.warning("Failed to read rejection learning file %s: %s", self.path, exc)

    def save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True, default=str))
        tmp.replace(self.path)

    async def sync_from_gateway(self, wallet: Any, limit: int = 50) -> int:
        records = await asyncio.to_thread(get_rejection_feedback, wallet, limit)
        added = self.ingest(records)
        if added:
            self.save()
        return added

    def ingest(self, records: Iterable[Dict[str, Any]]) -> int:
        processed = set(self.state.get("processed_ids", []))
        added = 0

        for rec in records or []:
            rec_id = str(rec.get("id") or rec.get("prospect_id") or "")
            if not rec_id or rec_id in processed:
                continue

            lead = rec.get("lead_snapshot") or {}
            summary = rec.get("rejection_summary") or {}
            reasons = self._extract_reasons(summary)

            for reason in reasons:
                self._inc("reason_counts", reason)

            domain = _domain_from_url(lead.get("website", ""))
            if domain:
                self._inc("domain_counts", _registrableish_domain(domain))

            role = self._role_key(lead.get("role", ""))
            if role:
                self._inc("role_counts", role)

            email_local = _lower(lead.get("email", "")).split("@", 1)[0]
            if email_local:
                self._inc("email_local_counts", email_local)

            self.state.setdefault("examples", []).append({
                "id": rec_id,
                "created_at": rec.get("created_at"),
                "epoch_number": rec.get("epoch_number"),
                "business": lead.get("business"),
                "email": lead.get("email"),
                "role": lead.get("role"),
                "domain": domain,
                "reasons": reasons[:8],
            })
            self.state["examples"] = self.state["examples"][-200:]

            processed.add(rec_id)
            added += 1

        self.state["processed_ids"] = sorted(processed)[-1000:]
        self.state["last_sync_at"] = datetime.now(timezone.utc).isoformat()
        return added

    def repair(self, lead: Dict[str, Any]) -> Dict[str, Any]:
        repaired = dict(lead)

        if "linkedin.com" in _lower(repaired.get("source_url")):
            repaired["source_url"] = repaired.get("website", "")
            repaired["source_type"] = "company_site"

        repaired["linkedin"] = self._canonical_linkedin(
            repaired.get("linkedin", ""), "https://linkedin.com/in/"
        )
        repaired["company_linkedin"] = self._canonical_linkedin(
            repaired.get("company_linkedin", ""), "https://linkedin.com/company/"
        )

        emp = _clean(repaired.get("employee_count"))
        emp_key = emp.replace(",", "").lower()
        if emp_key in EMPLOYEE_COUNT_ALIASES:
            repaired["employee_count"] = EMPLOYEE_COUNT_ALIASES[emp_key]

        if repaired.get("description") and len(repaired["description"]) > 500:
            repaired["description"] = repaired["description"][:500].rsplit(" ", 1)[0]

        return repaired

    def blocks(self, lead: Dict[str, Any]) -> List[str]:
        blocks: List[str] = []
        domain = _registrableish_domain(_domain_from_url(lead.get("website", "")))
        role = self._role_key(lead.get("role", ""))
        email_local = _lower(lead.get("email", "")).split("@", 1)[0]

        if self._count("domain_counts", domain) >= self.min_pattern_count:
            blocks.append("learned_rejected_domain")
        if self._count("role_counts", role) >= self.min_pattern_count and not LeadQualityMiner._is_decision_maker_role(role):
            blocks.append("learned_rejected_role")
        if self._count("email_local_counts", email_local) >= self.min_pattern_count:
            blocks.append("learned_rejected_email_pattern")
        return blocks

    def penalty(self, lead: Dict[str, Any]) -> int:
        penalty = 0
        domain = _registrableish_domain(_domain_from_url(lead.get("website", "")))
        role = self._role_key(lead.get("role", ""))

        if self._count("domain_counts", domain) > 0:
            penalty += min(20, 6 * self._count("domain_counts", domain))
        if self._count("role_counts", role) > 0:
            penalty += min(15, 5 * self._count("role_counts", role))
        return penalty

    def _inc(self, bucket: str, key: str) -> None:
        key = _lower(key)
        if not key:
            return
        self.state.setdefault(bucket, {})
        self.state[bucket][key] = int(self.state[bucket].get(key, 0)) + 1

    def _count(self, bucket: str, key: str) -> int:
        return int(self.state.get(bucket, {}).get(_lower(key), 0))

    @staticmethod
    def _role_key(role: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", _lower(role)).strip()

    @staticmethod
    def _canonical_linkedin(url: str, prefix: str) -> str:
        url = _ensure_https(url)
        if not url:
            return ""
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path = parsed.path.rstrip("/")
        if host == "www.linkedin.com":
            host = "linkedin.com"
        canonical = f"https://{host}{path}"
        return canonical if canonical.startswith(prefix) else url

    @staticmethod
    def _extract_reasons(summary: Dict[str, Any]) -> List[str]:
        reasons: List[str] = []
        failures = summary.get("common_failures") or summary.get("failures") or []
        if isinstance(failures, dict):
            failures = [failures]
        for failure in failures:
            if isinstance(failure, dict):
                check = failure.get("check_name") or failure.get("check") or ""
                stage = failure.get("stage") or ""
                msg = failure.get("message") or failure.get("reason") or ""
                raw = "|".join(p for p in [stage, check, msg] if p)
            else:
                raw = str(failure)
            key = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
            if key:
                reasons.append(key[:120])

        primary = summary.get("primary_rejection_reason") or summary.get("reason")
        if primary:
            key = re.sub(r"[^a-z0-9]+", "_", str(primary).lower()).strip("_")
            if key:
                reasons.append(key[:120])
        return sorted(set(reasons)) or ["unknown_rejection"]


class LeadQualityMiner:
    def __init__(
        self,
        wallet: Any,
        miner_hotkey: str,
        cache_path: str = "miner_state/submitted_hashes.json",
        suppression_path: str = "miner_state/suppression_list.json",
        feedback_path: str = "miner_state/rejection_learning.json",
        min_confidence: int = 85,
    ):
        self.wallet = wallet
        self.miner_hotkey = miner_hotkey
        self.min_confidence = min_confidence
        self.submitted_hashes = LocalJsonSet(Path(cache_path))
        self.suppression = SuppressionList(Path(suppression_path))
        self.feedback = RejectionFeedbackLearner(Path(feedback_path))
        self.truelist_key = os.getenv("TRUELIST_API_KEY", "")

    async def run_once(
        self,
        count: int = 5,
        industry: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Tuple[int, int]:
        added_feedback = await self.feedback.sync_from_gateway(
            self.wallet,
            limit=int(os.getenv("MINER_REJECTION_FEEDBACK_LIMIT", "50")),
        )
        if added_feedback:
            logger.info("Ingested %s new validator rejection feedback record(s)", added_feedback)

        raw_leads = await get_leads(count, industry=industry, region=region)
        decisions = await self.prepare_leads(raw_leads)
        accepted = [d.lead for d in decisions if d.accepted and d.lead]

        submitted = 0
        for lead in accepted:
            if await self.submit_lead(lead):
                submitted += 1

        return len(accepted), submitted

    async def prepare_leads(self, raw_leads: Iterable[Dict[str, Any]]) -> List[LeadDecision]:
        candidates = [self.normalize_lead(raw) for raw in raw_leads]
        decisions = await asyncio.gather(*(self.evaluate_lead(c) for c in candidates))
        return list(decisions)

    def normalize_lead(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        first = _clean(raw.get("first") or raw.get("First"))
        last = _clean(raw.get("last") or raw.get("Last"))
        full_name = _clean(raw.get("full_name") or raw.get("Owner Full name"))
        if not full_name:
            full_name = " ".join(p for p in [first, last] if p)
        if not first and full_name:
            first = full_name.split()[0]
        if not last and len(full_name.split()) > 1:
            last = full_name.split()[-1]

        website = _ensure_https(raw.get("website") or raw.get("Website"))
        source_url = _ensure_https(raw.get("source_url") or raw.get("Source URL") or website)

        lead = {
            "business": _clean(raw.get("business") or raw.get("Business")),
            "full_name": full_name,
            "first": first,
            "last": last,
            "email": _lower(raw.get("email") or raw.get("Owner(s) Email")),
            "role": _clean(raw.get("role") or raw.get("Title")),
            "website": website,
            "industry": _clean(raw.get("industry") or raw.get("Industry")),
            "sub_industry": _clean(raw.get("sub_industry") or raw.get("Sub Industry")),
            "country": _clean(raw.get("country") or raw.get("Country") or "United States"),
            "state": _clean(raw.get("state") or raw.get("State")),
            "city": _clean(raw.get("city") or raw.get("City")),
            "linkedin": _ensure_https(raw.get("linkedin") or raw.get("LinkedIn")),
            "company_linkedin": _ensure_https(raw.get("company_linkedin") or raw.get("Company LinkedIn")),
            "source_url": source_url,
            "source_type": _clean(raw.get("source_type") or "company_site"),
            "description": _clean(raw.get("description") or raw.get("Description")),
            "employee_count": _clean(raw.get("employee_count") or raw.get("Employee Count")),
            "hq_country": _clean(raw.get("hq_country") or raw.get("country") or raw.get("Country") or "United States"),
            "hq_state": _clean(raw.get("hq_state") or raw.get("state") or raw.get("State")),
            "hq_city": _clean(raw.get("hq_city") or raw.get("city") or raw.get("City")),
            "phone_numbers": raw.get("phone_numbers", []),
            "socials": raw.get("socials", {}),
            "source": self.miner_hotkey,
            "confidence_score": 0,
            "miner_validation_ts": datetime.now(timezone.utc).isoformat(),
        }
        return self.feedback.repair(lead)

    async def evaluate_lead(self, lead: Dict[str, Any]) -> LeadDecision:
        reasons: List[str] = []

        for reason in self._hard_rule_failures(lead):
            reasons.append(reason)

        suppressed = self.suppression.blocks(lead)
        if suppressed:
            reasons.append(suppressed)

        reasons.extend(self.feedback.blocks(lead))

        h = _lead_hash(lead)
        if self.submitted_hashes.contains(h):
            reasons.append("local_duplicate")

        if not reasons:
            website_ok = await self.verify_website(lead["website"])
            if not website_ok:
                reasons.append("website_inaccessible")

        if not reasons:
            email_ok, status = await self.verify_email_quality(lead)
            if not email_ok:
                reasons.append(f"email_quality_{status}")

        if not reasons:
            if check_email_duplicate(lead["email"]):
                reasons.append("gateway_email_duplicate")
            elif lead.get("linkedin") and lead.get("company_linkedin"):
                if check_linkedin_combo_duplicate(lead["linkedin"], lead["company_linkedin"]):
                    reasons.append("gateway_linkedin_duplicate")

        score = self.confidence_score(lead, reasons)
        lead["confidence_score"] = score

        if reasons or score < self.min_confidence:
            logger.info(
                "SKIP lead business=%s email=%s score=%s reasons=%s",
                lead.get("business"),
                lead.get("email"),
                score,
                ",".join(reasons or ["low_confidence"]),
            )
            return LeadDecision(False, score, reasons or ["low_confidence"], lead)

        logger.info(
            "ACCEPT lead business=%s contact=%s email=%s score=%s",
            lead.get("business"),
            lead.get("full_name"),
            lead.get("email"),
            score,
        )
        return LeadDecision(True, score, [], lead)

    def _hard_rule_failures(self, lead: Dict[str, Any]) -> List[str]:
        failures: List[str] = []
        required = [
            "business",
            "full_name",
            "first",
            "last",
            "email",
            "role",
            "website",
            "industry",
            "sub_industry",
            "country",
            "city",
            "linkedin",
            "company_linkedin",
            "source_url",
            "description",
            "employee_count",
            "hq_country",
        ]
        missing = [field for field in required if not _clean(lead.get(field))]
        if missing:
            failures.append(f"missing_{'_'.join(missing[:3])}")

        if not self._valid_email_syntax(lead.get("email", "")):
            failures.append("invalid_email_format")
        if self._is_generic_email(lead.get("email", "")):
            failures.append("generic_email")
        if not self._email_matches_company_domain(lead):
            failures.append("email_domain_mismatch")
        if not self._email_contains_name(lead):
            failures.append("email_name_mismatch")
        if not self._is_decision_maker_role(lead.get("role", "")):
            failures.append("non_decision_maker_role")
        if not _lower(lead.get("linkedin")).startswith("https://linkedin.com/in/"):
            failures.append("invalid_person_linkedin")
        if not _lower(lead.get("company_linkedin")).startswith("https://linkedin.com/company/"):
            failures.append("invalid_company_linkedin")
        if "linkedin.com" in _lower(lead.get("source_url")):
            failures.append("source_url_linkedin_blocked")
        if lead.get("employee_count") and lead["employee_count"] not in VALID_EMPLOYEE_COUNTS:
            failures.append("invalid_employee_count")
        if _lower(lead.get("country")) in {"united states", "usa", "us"} and not _clean(lead.get("state")):
            failures.append("missing_us_state")
        if _lower(lead.get("hq_country")) in {"united states", "usa", "us"} and not _clean(lead.get("hq_state")):
            failures.append("missing_us_hq_state")

        return failures

    @staticmethod
    def _valid_email_syntax(email: str) -> bool:
        return bool(re.match(r"^[a-z0-9._%+\-']+@[a-z0-9.\-]+\.[a-z]{2,}$", _lower(email)))

    @staticmethod
    def _is_generic_email(email: str) -> bool:
        local = _lower(email).split("@", 1)[0]
        return local in GENERIC_EMAIL_PREFIXES

    @staticmethod
    def _email_contains_name(lead: Dict[str, Any]) -> bool:
        local = _lower(lead.get("email")).split("@", 1)[0]
        local_compact = re.sub(r"[^a-z]", "", local)
        first = re.sub(r"[^a-z]", "", _lower(lead.get("first")))
        last = re.sub(r"[^a-z]", "", _lower(lead.get("last")))
        return bool((first and first in local_compact) or (last and last in local_compact))

    @staticmethod
    def _email_matches_company_domain(lead: Dict[str, Any]) -> bool:
        email_domain = _registrableish_domain(_email_domain(lead.get("email", "")))
        site_domain = _registrableish_domain(_domain_from_url(lead.get("website", "")))
        return bool(email_domain and site_domain and email_domain == site_domain)

    @staticmethod
    def _is_decision_maker_role(role: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", " ", _lower(role)).strip()
        return any(r == normalized or r in normalized for r in DECISION_MAKER_ROLES)

    async def verify_website(self, website: str) -> bool:
        if not website:
            return False
        if httpx is None:
            return False
        async with httpx.AsyncClient(follow_redirects=True, timeout=12) as client:
            try:
                resp = await client.head(website)
                if resp.status_code < 400:
                    return True
            except Exception:
                pass
            try:
                resp = await client.get(website)
                return resp.status_code < 400
            except Exception:
                return False

    async def verify_email_quality(self, lead: Dict[str, Any]) -> Tuple[bool, str]:
        if not self.truelist_key:
            return True, "syntax_only"
        if httpx is None:
            return False, "httpx_missing"

        email = lead["email"]
        url = f"https://api.truelist.io/api/v1/verify_inline?email={email}"
        async with httpx.AsyncClient(timeout=20) as client:
            try:
                resp = await client.post(url, headers={"Authorization": f"Bearer {self.truelist_key}"})
                if resp.status_code != 200:
                    return False, f"truelist_http_{resp.status_code}"
                data = resp.json()
                item = data.get("emails", [data])[0] if isinstance(data, dict) else {}
                state = _lower(item.get("email_state"))
                sub_state = _lower(item.get("email_sub_state"))
                status = sub_state or state or "unknown"
                if "email_ok" in {state, sub_state}:
                    return True, status
                return False, status
            except Exception as exc:
                return False, f"truelist_error_{type(exc).__name__}"

    def confidence_score(self, lead: Dict[str, Any], reasons: List[str]) -> int:
        if reasons:
            return 0

        score = 70
        score += 8 if self._email_contains_name(lead) else 0
        score += 7 if self._email_matches_company_domain(lead) else 0
        score += 5 if self._is_decision_maker_role(lead.get("role", "")) else 0
        score += 4 if lead.get("source_url") and lead.get("source_type") else 0
        score += 3 if len(lead.get("description", "")) >= 30 else 0
        score += 2 if lead.get("hq_city") and lead.get("hq_country") else 0
        score += 1 if lead.get("employee_count") in VALID_EMPLOYEE_COUNTS else 0
        score -= self.feedback.penalty(lead)
        return max(0, min(100, score))

    async def submit_lead(self, lead: Dict[str, Any]) -> bool:
        """Clean miner-side integration point for gateway submission."""
        h = _lead_hash(lead)
        if self.submitted_hashes.contains(h):
            logger.info("SKIP submit local duplicate hash=%s", h[:12])
            return False

        presign = await asyncio.to_thread(gateway_get_presigned_url, self.wallet, lead)
        if not presign:
            logger.warning("Gateway presign failed for %s", lead.get("email"))
            return False

        uploaded = await asyncio.to_thread(gateway_upload_lead, presign["s3_url"], lead)
        if not uploaded:
            logger.warning("Gateway upload failed for %s", lead.get("email"))
            return False

        verified = await asyncio.to_thread(gateway_verify_submission, self.wallet, presign["lead_id"])
        if not verified:
            logger.warning("Gateway verification failed for %s", lead.get("email"))
            return False

        self.submitted_hashes.add(h)
        logger.info(
            "SUBMITTED lead business=%s email=%s lead_id=%s confidence=%s",
            lead.get("business"),
            lead.get("email"),
            presign["lead_id"],
            lead.get("confidence_score"),
        )
        return True
