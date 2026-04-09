"""
Fulfillment consensus computation (v_trust x stake weighted).

Mirrors compute_weighted_consensus from gateway/utils/consensus.py.
"""

import logging
from typing import List, Dict, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


def _get_supabase():
    from gateway.db.client import get_write_client
    return get_write_client()


async def _fetch_request_scores(request_id: str) -> List[dict]:
    supabase = _get_supabase()
    resp = supabase.table("fulfillment_scores") \
        .select("*") \
        .eq("request_id", request_id) \
        .execute()
    return resp.data or []


async def _fetch_validator_metagraph_data(validator_hotkeys: Set[str]) -> Dict[str, dict]:
    """Fetch v_trust and stake for validators from metagraph snapshots."""
    if not validator_hotkeys:
        return {}
    try:
        supabase = _get_supabase()
        resp = supabase.table("metagraph") \
            .select("hotkey, validator_trust, stake") \
            .in_("hotkey", list(validator_hotkeys)) \
            .execute()
        result = {}
        for row in (resp.data or []):
            result[row["hotkey"]] = {
                "v_trust": float(row.get("validator_trust", 0.0) or 0.0),
                "stake": float(row.get("stake", 0.0) or 0.0),
            }
        return result
    except Exception as e:
        logger.warning(f"Failed to fetch metagraph data: {e}")
        return {hk: {"v_trust": 1.0, "stake": 1.0} for hk in validator_hotkeys}


def _group_scores_by_lead(scores: List[dict]) -> Dict[tuple, List[dict]]:
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for s in scores:
        key = (s["submission_id"], s["miner_hotkey"], s["lead_id"])
        groups[key].append(s)
    return groups


async def compute_fulfillment_consensus(request_id: str) -> List[dict]:
    """
    Compute v_trust x stake weighted consensus for all leads in a request.

    Per lead:
    1. Tier 1 unanimous gate: any validator failing -> reject
    2. Tier 2 stake-weighted gate: >50% weighted pass required
    3. Tier 3 stake-weighted scoring for leads passing both gates
    """
    scores = await _fetch_request_scores(request_id)
    if not scores:
        return []

    validator_hotkeys = {s["validator_hotkey"] for s in scores}
    metagraph_data = await _fetch_validator_metagraph_data(validator_hotkeys)

    lead_groups = _group_scores_by_lead(scores)

    consensus_results = []
    for lead_key, validator_scores in lead_groups.items():
        submission_id, miner_hotkey, lead_id = lead_key

        weighted_scores = []
        for vs in validator_scores:
            hotkey = vs["validator_hotkey"]
            meta = metagraph_data.get(hotkey, {"v_trust": 1.0, "stake": 1.0})
            v_trust = float(meta.get("v_trust", 0.0))
            stake = float(meta.get("stake", 0.0))
            weight = v_trust * stake
            if weight > 0:
                weighted_scores.append({**vs, "weight": weight})

        if not weighted_scores:
            continue

        total_weight = sum(ws["weight"] for ws in weighted_scores)

        # Tier 1: unanimous hard-gate
        if any(not ws.get("tier1_passed") for ws in weighted_scores):
            consensus_results.append({
                "request_id": request_id,
                "submission_id": submission_id,
                "miner_hotkey": miner_hotkey,
                "lead_id": lead_id,
                "num_validators": len(weighted_scores),
                "consensus_intent_signal_final": 0.0,
                "consensus_final_score": 0.0,
                "consensus_tier2_passed": False,
                "any_fabricated": any(ws.get("all_fabricated") for ws in weighted_scores),
            })
            continue

        # Tier 2: stake-weighted gate
        weighted_tier2_pass = sum(
            ws["weight"] for ws in weighted_scores if ws.get("tier2_passed")
        ) / total_weight

        if weighted_tier2_pass <= 0.5:
            consensus_results.append({
                "request_id": request_id,
                "submission_id": submission_id,
                "miner_hotkey": miner_hotkey,
                "lead_id": lead_id,
                "num_validators": len(weighted_scores),
                "consensus_intent_signal_final": 0.0,
                "consensus_final_score": 0.0,
                "consensus_tier2_passed": False,
                "any_fabricated": any(ws.get("all_fabricated") for ws in weighted_scores),
            })
            continue

        # Tier 3: stake-weighted scoring
        passing_scores = [ws for ws in weighted_scores if ws.get("tier2_passed")]
        passing_weight = sum(ws["weight"] for ws in passing_scores)

        consensus_intent = sum(
            float(ws.get("intent_signal_final", 0) or 0) * ws["weight"]
            for ws in passing_scores
        ) / passing_weight

        consensus_final = sum(
            float(ws.get("final_score", 0) or 0) * ws["weight"]
            for ws in passing_scores
        ) / passing_weight

        consensus_results.append({
            "request_id": request_id,
            "submission_id": submission_id,
            "miner_hotkey": miner_hotkey,
            "lead_id": lead_id,
            "num_validators": len(weighted_scores),
            "consensus_intent_signal_final": round(consensus_intent, 4),
            "consensus_final_score": round(consensus_final, 4),
            "consensus_tier2_passed": True,
            "any_fabricated": any(ws.get("all_fabricated") for ws in weighted_scores),
        })

    return consensus_results
