#!/usr/bin/env python3
"""
Offline dry run for the miner-side LeadQualityMiner.

This script does not submit leads, call the gateway, call TrueList, or check
websites. It exercises local normalization, strict rule checks, suppression,
rejection-learning penalties, and confidence scoring.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miner_models.lead_quality_miner import LeadQualityMiner


class DryRunHotkey:
    ss58_address = "dry_run_hotkey"


class DryRunWallet:
    hotkey = DryRunHotkey()


class OfflineLeadQualityMiner(LeadQualityMiner):
    async def verify_website(self, website: str) -> bool:
        return bool(website)

    async def verify_email_quality(self, lead: Dict[str, Any]):
        return True, "offline"

    async def evaluate_lead(self, lead: Dict[str, Any]):
        # Keep the normal local checks, but skip gateway duplicate checks by
        # temporarily replacing the expensive branch with local-only logic.
        reasons: List[str] = []

        for reason in self._hard_rule_failures(lead):
            reasons.append(reason)

        suppressed = self.suppression.blocks(lead)
        if suppressed:
            reasons.append(suppressed)

        reasons.extend(self.feedback.blocks(lead))

        from miner_models.lead_quality_miner import _lead_hash

        if self.submitted_hashes.contains(_lead_hash(lead)):
            reasons.append("local_duplicate")

        score = self.confidence_score(lead, reasons)
        lead["confidence_score"] = score

        from miner_models.lead_quality_miner import LeadDecision

        if reasons or score < self.min_confidence:
            return LeadDecision(False, score, reasons or ["low_confidence"], lead)
        return LeadDecision(True, score, [], lead)


def sample_leads() -> List[Dict[str, Any]]:
    return [
        {
            "business": "Acme AI",
            "full_name": "Jane Smith",
            "first": "Jane",
            "last": "Smith",
            "email": "jane.smith@acmeai.com",
            "role": "CEO",
            "website": "https://acmeai.com",
            "industry": "Software",
            "sub_industry": "Enterprise Software",
            "country": "United States",
            "state": "California",
            "city": "San Francisco",
            "linkedin": "https://linkedin.com/in/janesmith",
            "company_linkedin": "https://linkedin.com/company/acme-ai",
            "source_url": "https://acmeai.com/about",
            "description": "B2B software company building AI workflow automation for enterprise teams.",
            "employee_count": "51-200",
            "hq_country": "United States",
            "hq_state": "California",
            "hq_city": "San Francisco",
        },
        {
            "business": "Bad Example",
            "full_name": "Support Team",
            "first": "Support",
            "last": "Team",
            "email": "info@badexample.com",
            "role": "Assistant",
            "website": "https://badexample.com",
            "industry": "Software",
            "sub_industry": "Enterprise Software",
            "country": "United States",
            "state": "California",
            "city": "San Francisco",
            "linkedin": "https://linkedin.com/in/supportteam",
            "company_linkedin": "https://linkedin.com/company/badexample",
            "source_url": "https://badexample.com",
            "description": "Example company.",
            "employee_count": "51-200",
            "hq_country": "United States",
            "hq_state": "California",
            "hq_city": "San Francisco",
        },
    ]


def load_leads(path: str | None) -> List[Dict[str, Any]]:
    if not path:
        return sample_leads()

    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("leads", [])
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array or an object with a 'leads' array")
    return data


async def main() -> int:
    parser = argparse.ArgumentParser(description="Offline dry-run miner lead quality checks")
    parser.add_argument("--input", help="Path to JSON array of leads. Uses built-in samples if omitted.")
    parser.add_argument("--min-confidence", type=int, default=85)
    parser.add_argument("--cache", default="miner_state/dry_run_submitted_hashes.json")
    parser.add_argument("--suppression", default="miner_state/suppression_list.json")
    parser.add_argument("--feedback", default="miner_state/rejection_learning.json")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    miner = OfflineLeadQualityMiner(
        wallet=DryRunWallet(),
        miner_hotkey=DryRunHotkey.ss58_address,
        cache_path=args.cache,
        suppression_path=args.suppression,
        feedback_path=args.feedback,
        min_confidence=args.min_confidence,
    )

    raw_leads = load_leads(args.input)
    normalized = [miner.normalize_lead(lead) for lead in raw_leads]
    decisions = await miner.prepare_leads(normalized)

    output = []
    for idx, decision in enumerate(decisions, 1):
        lead = decision.lead or {}
        output.append({
            "index": idx,
            "accepted": decision.accepted,
            "confidence_score": decision.confidence_score,
            "reasons": decision.reasons,
            "business": lead.get("business"),
            "full_name": lead.get("full_name"),
            "email": lead.get("email"),
            "role": lead.get("role"),
            "website": lead.get("website"),
        })

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        accepted = sum(1 for row in output if row["accepted"])
        print(f"Offline dry run complete: accepted={accepted}, skipped={len(output) - accepted}")
        for row in output:
            status = "ACCEPT" if row["accepted"] else "SKIP"
            print(f"\n[{status}] #{row['index']} {row['business']} - {row['full_name']}")
            print(f"  email: {row['email']}")
            print(f"  role: {row['role']}")
            print(f"  score: {row['confidence_score']}")
            print(f"  reasons: {', '.join(row['reasons']) if row['reasons'] else '-'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
