"""
Pydantic models for the lead fulfillment system.

FulfillmentICP and FulfillmentLead constrain industry/sub_industry/role_type
to canonical taxonomy values so Tier 1 ICP Fit Gate checks are free
deterministic equality comparisons.
"""

import re
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from gateway.qualification.models import (
    IntentSignal,
    IntentSignalSource,  # noqa: F401 — re-exported for convenience
    Seniority,
    LeadOutput,
    ICPPrompt,
)
try:
    from gateway.utils.industry_taxonomy import INDUSTRY_TAXONOMY
except ImportError:
    from validator_models.industry_taxonomy import INDUSTRY_TAXONOMY

# ---------------------------------------------------------------------------
# Taxonomy constraint sets (derived from the canonical taxonomy)
# ---------------------------------------------------------------------------
# Keys are sub-industries; values contain parent industry lists.
VALID_SUB_INDUSTRIES: set = set(INDUSTRY_TAXONOMY.keys())
VALID_INDUSTRIES: set = {
    ind for entry in INDUSTRY_TAXONOMY.values() for ind in entry["industries"]
}
SUB_INDUSTRY_TO_PARENTS: dict = {
    sub: entry["industries"] for sub, entry in INDUSTRY_TAXONOMY.items()
}

VALID_ROLE_TYPES: set = {
    "C-Level Executive", "VP", "Director", "Manager",
    "Sales", "Marketing", "Engineering", "Product",
    "Operations", "Finance", "HR", "Legal",
    "IT", "Customer Success", "Business Development",
    "Data & Analytics", "Design", "Research",
    "Supply Chain", "Consulting", "Other",
}


# ---------------------------------------------------------------------------
# Company-name scrubbing helper
# ---------------------------------------------------------------------------

COMPANY_PLACEHOLDER = "[company_name]"


def scrub_company_name(text: str, company: str) -> str:
    """Replace every whole-word occurrence of ``company`` in ``text`` with
    ``[company_name]``, case-insensitive.

    Used by the fulfillment request ingestion path so miners never see the
    identity of the client who submitted the request.  Possessive forms
    (``AcmeCorp's``) are preserved because the ``\\b`` word boundary ends
    before the apostrophe, so the ``'s`` stays attached to the placeholder
    (``[company_name]'s``).

    Short or generic company names (e.g. ``"Apple"``) may match unrelated
    occurrences of the same token elsewhere in the text.  That is an
    accepted trade-off; callers should supply a distinctive name.

    Returns the input unchanged when either ``text`` or ``company`` is
    empty.  Never raises.
    """
    if not text or not company:
        return text or ""

    pattern = re.compile(r"\b" + re.escape(company) + r"\b", re.IGNORECASE)
    scrubbed = pattern.sub(COMPANY_PLACEHOLDER, text)
    # Collapse any accidental whitespace runs that could have been created
    # if the original text had odd spacing around the name.
    scrubbed = re.sub(r"[ \t]{2,}", " ", scrubbed).strip()
    return scrubbed


# ---------------------------------------------------------------------------
# FulfillmentICP
# ---------------------------------------------------------------------------

class FulfillmentICP(BaseModel):
    """ICP published to miners for a fulfillment request."""

    icp_id: str = Field(default="")
    prompt: str = Field(..., min_length=1)
    industry: str = ""
    sub_industry: str = ""
    target_role_types: List[str] = Field(default_factory=list)
    target_roles: List[str] = Field(default_factory=list)
    target_seniority: str = ""
    employee_count: str = ""
    company_stage: str = ""
    geography: str = ""
    country: str = ""
    product_service: str = ""
    intent_signals: List[str] = Field(default_factory=list)
    num_leads: int = 10
    window_end: Optional[str] = None
    reveal_window_end: Optional[str] = None

    # Internal-only label for client identification in Supabase dashboards
    # (e.g. "Edward Burrowes 1").  Persisted to the dedicated `internal_label`
    # column on `fulfillment_requests`, NOT inside `icp_details`, so it is
    # never returned by /fulfillment/requests/active and miners never see it.
    # exclude=True makes model_dump() drop it, so it can't leak into the
    # hash/jsonb by accident.
    internal_label: str = Field(default="", exclude=True)

    # Client company name (e.g. "AcmeCorp").  REQUIRED.  Stored in the
    # dedicated `company` column on fulfillment_requests.  The gateway's
    # create_request endpoint additionally scrubs every occurrence of this
    # string from the free-text ICP fields (prompt, product_service,
    # intent_signals, target_roles) before persisting, replacing each match
    # with "[company_name]" so miners can never learn which client made the
    # request.  Like internal_label, Field(exclude=True) guarantees it never
    # reaches model_dump() -> icp_details -> miners.
    company: str = Field(..., min_length=1, exclude=True)

    @field_validator("industry")
    @classmethod
    def validate_industry(cls, v: str) -> str:
        if not v:
            return v
        if v not in VALID_INDUSTRIES:
            raise ValueError(f"Industry '{v}' not in taxonomy. Valid: {sorted(VALID_INDUSTRIES)}")
        return v

    @field_validator("sub_industry")
    @classmethod
    def validate_sub_industry(cls, v: str, info) -> str:
        if not v:
            return v
        if v not in VALID_SUB_INDUSTRIES:
            raise ValueError(f"Sub-industry '{v}' not in taxonomy")
        industry = info.data.get("industry", "")
        if industry and v not in SUB_INDUSTRY_TO_PARENTS.get(v, []) and industry not in INDUSTRY_TAXONOMY.get(v, {}).get("industries", []):
            raise ValueError(f"Sub-industry '{v}' does not belong to industry '{industry}'")
        return v

    @field_validator("target_role_types")
    @classmethod
    def validate_role_types(cls, v: List[str]) -> List[str]:
        invalid = [r for r in v if r not in VALID_ROLE_TYPES]
        if invalid:
            raise ValueError(f"Invalid role types: {invalid}. Valid: {sorted(VALID_ROLE_TYPES)}")
        return v

    @field_validator("employee_count")
    @classmethod
    def validate_employee_count(cls, v: str) -> str:
        if not v:
            return v
        v = v.strip()
        if re.match(r"^\d+$", v):
            return v
        if re.match(r"^\d+-\d+$", v):
            lo, hi = v.split("-")
            if int(lo) > int(hi):
                raise ValueError(f"Invalid range: {v}")
            return v
        if re.match(r"^\d+\+$", v):
            return v
        raise ValueError(f"Invalid employee_count format: '{v}'. Use '50-200', '500+', or '1000'")

    def to_icp_prompt(self) -> ICPPrompt:
        """Convert to ICPPrompt for scoring functions."""
        roles = self.target_roles or self.target_role_types
        return ICPPrompt(
            icp_id=self.icp_id,
            prompt=self.prompt,
            industry=self.industry,
            sub_industry=self.sub_industry,
            target_roles=roles,
            target_seniority=self.target_seniority,
            employee_count=self.employee_count,
            company_stage=self.company_stage,
            geography=self.geography,
            country=self.country,
            product_service=self.product_service,
            intent_signals=self.intent_signals,
        )


# ---------------------------------------------------------------------------
# FulfillmentLead
# ---------------------------------------------------------------------------

class FulfillmentLead(BaseModel):
    """Lead schema with PII — used in fulfillment commit-reveal.

    All fields are required except ``phone``.  Miners that submit
    sparse leads will be rejected at parse time rather than silently
    scoring zero.
    """

    # PII fields (included in hash, stripped by to_lead_output)
    full_name: str
    email: str
    linkedin_url: str
    phone: str = ""

    # Company info
    business: str
    company_linkedin: str
    company_website: str
    employee_count: str

    # Company HQ location (used for ICP country/state matching)
    company_hq_country: str
    company_hq_state: str
    company_hq_city: str = ""

    # Industry
    industry: str
    sub_industry: str

    # Company description (free-form, written by the miner).
    # REQUIRED — this flows into the validator's Stage 5 classification
    # pipeline (validator_models/stage5_verification.py::classify_company_industry),
    # which performs a 3-stage check:
    #   1. Compare miner description against scraped website/LinkedIn content
    #      (INVALID → stage1_invalid_description → reject)
    #   2. Embed the refined description
    #   3. LLM ranks top-3 industry/sub_industry pairs
    # If the description is missing or doesn't match the website, Stage 5
    # rejects the lead BEFORE intent scoring runs, the same way sourcing
    # rejects leads with bad descriptions.
    description: str = Field(..., min_length=30)

    # Contact location
    country: str
    city: str
    state: str

    # Role
    role: str
    role_type: str
    seniority: str

    # Intent
    intent_signals: List[IntentSignal] = Field(..., min_length=1)

    @field_validator("industry")
    @classmethod
    def validate_industry(cls, v: str) -> str:
        if v not in VALID_INDUSTRIES:
            raise ValueError(f"Industry '{v}' not in taxonomy. Valid: {sorted(VALID_INDUSTRIES)}")
        return v

    @field_validator("sub_industry")
    @classmethod
    def validate_sub_industry(cls, v: str) -> str:
        if v not in VALID_SUB_INDUSTRIES:
            raise ValueError(f"Sub-industry '{v}' not in taxonomy. Valid: {sorted(VALID_SUB_INDUSTRIES)}")
        return v

    @field_validator("role_type")
    @classmethod
    def validate_role_type(cls, v: str) -> str:
        if v not in VALID_ROLE_TYPES:
            raise ValueError(f"Role type '{v}' not valid. Valid: {sorted(VALID_ROLE_TYPES)}")
        return v

    def to_lead_output(self) -> LeadOutput:
        """Strip PII and convert to LeadOutput for scoring functions.

        LeadOutput's country/state are company-level fields, so we map
        from the HQ fields here (contact-level country/state stay on
        the FulfillmentLead only).
        """
        seniority_value = self.seniority
        seniority_map = {"Senior": "Manager"}
        if seniority_value in seniority_map:
            seniority_value = seniority_map[seniority_value]

        return LeadOutput(
            lead_id=0,
            business=self.business,
            company_linkedin=self.company_linkedin,
            company_website=self.company_website,
            employee_count=self.employee_count,
            industry=self.industry,
            sub_industry=self.sub_industry,
            country=self.company_hq_country,
            city=self.city,
            state=self.company_hq_state,
            role=self.role,
            role_type=self.role_type,
            seniority=seniority_value,
            intent_signals=self.intent_signals,
        )

    def to_validator_dict(self) -> dict:
        """Convert to dict with keys expected by validator_models check functions.

        The validator extraction utilities (get_website, get_linkedin,
        get_first_name, get_last_name, etc.) expect specific key names
        that differ from FulfillmentLead fields.  The returned dict is
        intentionally mutable — Stage 0-2 checks add fields like
        ``domain_age_days``, ``has_mx``, etc. in-place, and Stage 4-5
        reads them back.
        """
        d = self.model_dump(exclude={"intent_signals"})
        d["website"] = self.company_website
        d["linkedin"] = self.linkedin_url
        d["hq_country"] = self.company_hq_country
        d["hq_state"] = self.company_hq_state
        d["hq_city"] = self.company_hq_city
        parts = self.full_name.strip().split(None, 1)
        d["first"] = parts[0] if parts else ""
        d["last"] = parts[1] if len(parts) > 1 else ""
        return d


# ---------------------------------------------------------------------------
# Commit / Reveal request models
# ---------------------------------------------------------------------------

class CommitHashEntry(BaseModel):
    """Single lead hash submitted during commit (no lead_id yet)."""
    hash: str


class LeadHashEntry(BaseModel):
    """Lead hash with gateway-assigned ID (stored after commit)."""
    lead_id: str
    hash: str


class FulfillmentCommitRequest(BaseModel):
    """Miner commit payload — hashes only, no lead data."""
    request_id: str
    miner_hotkey: str
    lead_hashes: List[CommitHashEntry]
    schema_version: int
    signature: str
    timestamp: int
    nonce: str


class FulfillmentRevealRequest(BaseModel):
    """Miner reveal payload — full lead data."""
    request_id: str
    submission_id: str
    miner_hotkey: str
    leads: List[FulfillmentLead]
    signature: str
    timestamp: int
    nonce: str


# ---------------------------------------------------------------------------
# Score result
# ---------------------------------------------------------------------------

class FulfillmentScoreResult(BaseModel):
    """Per-lead, per-validator score result."""
    lead_id: str = ""
    tier1_passed: bool = False
    tier2_passed: bool = False
    email_verified: bool = False
    person_verified: bool = False
    company_verified: bool = False
    rep_score: float = 0.0
    intent_signal_raw: float = 0.0
    intent_signal_final: float = 0.0
    intent_decay_multiplier: float = 0.0
    final_score: float = 0.0
    all_fabricated: bool = False
    failure_reason: Optional[str] = None
    # Per-miner-signal breakdown, only populated when Tier 3 intent scoring runs.
    # Each entry maps a miner-submitted signal to the best-matching client
    # (ICP) intent signal, plus the raw/after-decay score for that signal.
    # Fields per entry:
    #   url, description, snippet, date, source
    #   raw_score, after_decay_score, decay_multiplier, confidence, date_status
    #   matched_icp_signal_idx (int, -1 if no match)
    #   matched_icp_signal (str or None)
    intent_signals_detail: List[dict] = Field(default_factory=list)
