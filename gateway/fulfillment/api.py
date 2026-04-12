"""
Fulfillment API Router

7 endpoints for the lead fulfillment commit-reveal system.
"""

import os
import logging
import base64
import time as _time
from uuid import uuid4
from datetime import datetime, timezone, timedelta
from typing import List

from fastapi import APIRouter, HTTPException

from gateway.fulfillment.config import T_EPOCHS, T_SECONDS_OVERRIDE, M_MINUTES, FULFILLMENT_BANS_ENABLED, epochs_to_seconds
from gateway.fulfillment.hashing import HASH_SCHEMA_VERSION, hash_request, verify_commit
from gateway.fulfillment.models import (
    FulfillmentICP,
    FulfillmentLead,
    FulfillmentCommitRequest,
    FulfillmentRevealRequest,
    LeadHashEntry,
    FulfillmentScoreResult,
)
from gateway.models.events import EventType
from gateway.utils.bans import is_hotkey_banned, ban_hotkey

logger = logging.getLogger(__name__)

_SIG_TIMESTAMP_TOLERANCE = 300  # 5 minutes

fulfillment_router = APIRouter(prefix="/fulfillment", tags=["fulfillment"])


def _enable_fulfillment() -> bool:
    return os.getenv("ENABLE_FULFILLMENT", "false").lower() == "true"


def _get_supabase():
    from gateway.db.client import get_write_client
    return get_write_client()


def _get_tempo(supabase) -> int:
    """Fetch current subnet tempo from DB, default 360."""
    try:
        resp = supabase.table("subnet_state") \
            .select("tempo") \
            .limit(1) \
            .execute()
        if resp.data:
            return int(resp.data[0].get("tempo", 360))
    except Exception:
        pass
    return 360


def _log_event(event_type: EventType, payload: dict) -> None:
    """Best-effort transparency log insert."""
    from gateway.config import BITTENSOR_NETWORK

    if BITTENSOR_NETWORK == "test":
        logger.info(
            f"⚠️ TESTNET MODE: Skipping {event_type.value} log to protect production transparency_log"
        )
        return

    try:
        supabase = _get_supabase()
        supabase.table("transparency_log").insert({
            "event_type": event_type.value,
            "payload": payload,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logger.warning(f"Failed to log {event_type.value}: {e}")


def _verify_fulfillment_signature(
    event_type: str, hotkey: str, request_id: str,
    signature: str, nonce: str, timestamp: int,
) -> None:
    """Verify a miner's Ed25519 signature on a fulfillment request.
    The signed message binds to the request_id to prevent replay across requests.
    Raises HTTPException(403) on failure."""
    now_ts = int(_time.time())
    if abs(now_ts - timestamp) > _SIG_TIMESTAMP_TOLERANCE:
        raise HTTPException(403, detail="Timestamp too old or too far in the future")

    msg = f"{event_type}:{hotkey}:{request_id}:{nonce}:{timestamp}"
    try:
        from bittensor import Keypair
        sig_bytes = base64.b64decode(signature)
        kp = Keypair(ss58_address=hotkey)
        if not kp.verify(msg, sig_bytes):
            raise HTTPException(403, detail="Invalid signature")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Signature verification error: {e}")
        raise HTTPException(403, detail="Signature verification failed")


# ---------------------------------------------------------------
# POST /fulfillment/request  — client creates a new ICP request
# ---------------------------------------------------------------
@fulfillment_router.post("/request")
async def create_request(icp: FulfillmentICP):
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled on this gateway")

    supabase = _get_supabase()
    now = datetime.now(timezone.utc)
    request_id = str(uuid4())

    if T_SECONDS_OVERRIDE > 0:
        commit_seconds = T_SECONDS_OVERRIDE
    else:
        tempo = _get_tempo(supabase)
        commit_seconds = epochs_to_seconds(T_EPOCHS, tempo)
    window_end = now + timedelta(seconds=commit_seconds)
    reveal_window_end = window_end + timedelta(minutes=M_MINUTES)
    icp_dict = icp.model_dump(mode="json")
    req_hash = hash_request(icp_dict)

    supabase.table("fulfillment_requests").insert({
        "request_id": request_id,
        "request_hash": req_hash,
        "icp_details": icp_dict,
        "num_leads": icp.num_leads,
        "window_start": now.isoformat(),
        "window_end": window_end.isoformat(),
        "reveal_window_end": reveal_window_end.isoformat(),
        "status": "open",
        "created_by": "api",
    }).execute()

    _log_event(EventType.FULFILLMENT_REQUEST_CREATED, {
        "request_id": request_id,
        "request_hash": req_hash,
        "window_start": now.isoformat(),
        "window_end": window_end.isoformat(),
        "reveal_window_end": reveal_window_end.isoformat(),
    })

    return {
        "request_id": request_id,
        "request_hash": req_hash,
        "window_start": now.isoformat(),
        "window_end": window_end.isoformat(),
        "reveal_window_end": reveal_window_end.isoformat(),
        "num_leads": icp.num_leads,
    }


# ---------------------------------------------------------------
# GET /fulfillment/requests/active  — miners poll for open ICPs
# ---------------------------------------------------------------
@fulfillment_router.get("/requests/active")
async def get_active_requests(miner_hotkey: str = ""):
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled")

    supabase = _get_supabase()

    if miner_hotkey:
        banned, reason = await is_hotkey_banned(miner_hotkey)
        if banned:
            raise HTTPException(403, detail=f"Hotkey banned: {reason}")

    now = datetime.now(timezone.utc)
    resp = supabase.table("fulfillment_requests") \
        .select("*") \
        .eq("status", "open") \
        .gt("window_end", now.isoformat()) \
        .execute()

    requests_out = []
    for r in (resp.data or []):
        icp = r.get("icp_details", {})
        requests_out.append({
            "request_id": r["request_id"],
            "icp": icp,
            "num_leads": r["num_leads"],
            "window_end": r["window_end"],
            "reveal_window_end": r["reveal_window_end"],
        })

    return {
        "requests": requests_out,
        "gateway_server_time": now.isoformat(),
    }


# ---------------------------------------------------------------
# POST /fulfillment/commit  — miner submits lead hashes
# ---------------------------------------------------------------
@fulfillment_router.post("/commit")
async def commit_leads(commit: FulfillmentCommitRequest):
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled")

    _verify_fulfillment_signature(
        "FULFILLMENT_COMMIT", commit.miner_hotkey, commit.request_id,
        commit.signature, commit.nonce, commit.timestamp,
    )

    supabase = _get_supabase()

    banned, reason = await is_hotkey_banned(commit.miner_hotkey)
    if banned:
        raise HTTPException(403, detail=f"Hotkey banned: {reason}")

    if commit.schema_version != HASH_SCHEMA_VERSION:
        raise HTTPException(422, detail=(
            f"Schema version mismatch: client={commit.schema_version}, "
            f"server={HASH_SCHEMA_VERSION}. Update your miner."
        ))

    req_resp = supabase.table("fulfillment_requests") \
        .select("*") \
        .eq("request_id", commit.request_id) \
        .execute()
    if not req_resp.data:
        raise HTTPException(404, detail="Request not found")

    req = req_resp.data[0]
    if req["status"] != "open":
        raise HTTPException(400, detail=f"Window not open (status={req['status']})")

    now = datetime.now(timezone.utc)
    if now > datetime.fromisoformat(req["window_end"]):
        raise HTTPException(400, detail="Commit window expired")

    if len(commit.lead_hashes) > req["num_leads"]:
        raise HTTPException(422, detail=(
            f"Too many leads: submitted {len(commit.lead_hashes)}, max {req['num_leads']}"
        ))

    lead_hash_entries: List[dict] = []
    for entry in commit.lead_hashes:
        lead_hash_entries.append({
            "lead_id": str(uuid4()),
            "hash": entry.hash,
        })

    try:
        sub_resp = supabase.rpc("fulfillment_accept_commit", {
            "p_request_id": commit.request_id,
            "p_miner_hotkey": commit.miner_hotkey,
            "p_lead_hashes": lead_hash_entries,
        }).execute()
        submission_id = sub_resp.data
    except Exception as e:
        err_msg = str(e)
        if "unique" in err_msg.lower() or "duplicate" in err_msg.lower():
            existing = supabase.table("fulfillment_submissions") \
                .select("submission_id") \
                .eq("request_id", commit.request_id) \
                .eq("miner_hotkey", commit.miner_hotkey) \
                .execute()
            existing_id = existing.data[0]["submission_id"] if existing.data else None
            raise HTTPException(409, detail={
                "message": "Already committed for this request",
                "submission_id": existing_id,
            })
        raise HTTPException(500, detail=f"Commit failed: {err_msg}")

    _log_event(EventType.FULFILLMENT_COMMIT, {
        "request_id": commit.request_id,
        "submission_id": submission_id,
        "miner_hotkey": commit.miner_hotkey,
        "lead_hashes": lead_hash_entries,
        "submission_timestamp": now.isoformat(),
    })

    return {
        "submission_id": submission_id,
        "lead_ids": [e["lead_id"] for e in lead_hash_entries],
    }


# ---------------------------------------------------------------
# POST /fulfillment/reveal  — miner reveals lead data
# ---------------------------------------------------------------
@fulfillment_router.post("/reveal")
async def reveal_leads(reveal: FulfillmentRevealRequest):
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled")

    _verify_fulfillment_signature(
        "FULFILLMENT_REVEAL", reveal.miner_hotkey, reveal.request_id,
        reveal.signature, reveal.nonce, reveal.timestamp,
    )

    supabase = _get_supabase()

    banned, reason = await is_hotkey_banned(reveal.miner_hotkey)
    if banned:
        raise HTTPException(403, detail=f"Hotkey banned: {reason}")

    sub_resp = supabase.table("fulfillment_submissions") \
        .select("*") \
        .eq("submission_id", reveal.submission_id) \
        .eq("request_id", reveal.request_id) \
        .eq("miner_hotkey", reveal.miner_hotkey) \
        .execute()
    if not sub_resp.data:
        raise HTTPException(404, detail="Submission not found")

    submission = sub_resp.data[0]
    if submission["revealed"]:
        raise HTTPException(400, detail="Already revealed")

    req_resp = supabase.table("fulfillment_requests") \
        .select("window_end, reveal_window_end") \
        .eq("request_id", reveal.request_id) \
        .execute()
    if not req_resp.data:
        raise HTTPException(404, detail="Request not found")

    req = req_resp.data[0]
    now = datetime.now(timezone.utc)
    window_end_dt = datetime.fromisoformat(req["window_end"])
    reveal_end_dt = datetime.fromisoformat(req["reveal_window_end"])

    if now < window_end_dt:
        raise HTTPException(400, detail="Commit window still open — cannot reveal yet")
    if now > reveal_end_dt:
        raise HTTPException(400, detail="Reveal window expired")

    committed_hashes: list = submission["lead_hashes"]
    if len(reveal.leads) != len(committed_hashes):
        raise HTTPException(422, detail=(
            f"Must reveal all committed leads: expected {len(committed_hashes)}, got {len(reveal.leads)}"
        ))

    lead_data_list = []
    for i, lead in enumerate(reveal.leads):
        lead_dict = lead.model_dump(mode="json")
        committed_hash = committed_hashes[i]["hash"]
        if not verify_commit(committed_hash, lead_dict):
            raise HTTPException(
                400,
                detail=f"Hash mismatch for lead index {i} (lead_id={committed_hashes[i]['lead_id']})"
            )
        lead_data_list.append({
            "lead_id": committed_hashes[i]["lead_id"],
            "data": lead_dict,
        })

    supabase.table("fulfillment_submissions").update({
        "revealed": True,
        "revealed_at": now.isoformat(),
        "lead_data": lead_data_list,
    }).eq("submission_id", reveal.submission_id).execute()

    _log_event(EventType.FULFILLMENT_REVEAL, {
        "request_id": reveal.request_id,
        "miner_hotkey": reveal.miner_hotkey,
        "reveal_timestamp": now.isoformat(),
    })

    return {"status": "revealed", "num_leads": len(lead_data_list)}


# ---------------------------------------------------------------
# GET /fulfillment/scoring  — validators fetch revealed leads for scoring
# ---------------------------------------------------------------
@fulfillment_router.get("/scoring")
async def get_scoring_requests(validator_hotkey: str = ""):
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled")

    supabase = _get_supabase()

    resp = supabase.table("fulfillment_requests") \
        .select("*") \
        .eq("status", "scoring") \
        .execute()

    out = []
    for r in (resp.data or []):
        subs_resp = supabase.table("fulfillment_submissions") \
            .select("*") \
            .eq("request_id", r["request_id"]) \
            .eq("revealed", True) \
            .execute()

        submissions = []
        for s in (subs_resp.data or []):
            lead_data = s.get("lead_data") or []
            lead_hashes = s.get("lead_hashes") or []
            submissions.append({
                "submission_id": s["submission_id"],
                "miner_hotkey": s["miner_hotkey"],
                "leads": [entry.get("data", {}) for entry in lead_data],
                "lead_ids": [entry.get("lead_id", "") for entry in lead_hashes],
            })

        out.append({
            "request_id": r["request_id"],
            "icp": r.get("icp_details", {}),
            "status": r["status"],
            "submissions": submissions,
        })

    return {"requests": out}


# ---------------------------------------------------------------
# POST /fulfillment/score  — validator submits scores
# ---------------------------------------------------------------
@fulfillment_router.post("/score")
async def submit_scores(
    request_id: str,
    validator_hotkey: str,
    scores: List[dict],
):
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled")

    supabase = _get_supabase()

    for s in scores:
        if not s.get("request_id"):
            s["request_id"] = request_id

    try:
        supabase.rpc("fulfillment_upsert_scores", {
            "p_scores": scores,
            "p_validator_hotkey": validator_hotkey,
        }).execute()
    except Exception as e:
        raise HTTPException(500, detail=f"Score submission failed: {e}")

    _log_event(EventType.FULFILLMENT_SCORED, {
        "request_id": request_id,
        "scores": [
            {"miner_hotkey": s.get("miner_hotkey"), "lead_id": s.get("lead_id"),
             "score": s.get("final_score"), "reason": s.get("failure_reason")}
            for s in scores
        ],
        "score_timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {"status": "scores_accepted", "count": len(scores)}


# ---------------------------------------------------------------
# GET /fulfillment/results/{request_id}  — client fetches results
# ---------------------------------------------------------------
@fulfillment_router.get("/results/{request_id}")
async def get_results(request_id: str):
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled")

    supabase = _get_supabase()

    req_resp = supabase.table("fulfillment_requests") \
        .select("status, successor_request_id") \
        .eq("request_id", request_id) \
        .execute()
    if not req_resp.data:
        raise HTTPException(404, detail="Request not found")

    req = req_resp.data[0]

    consensus_resp = supabase.table("fulfillment_score_consensus") \
        .select("*") \
        .eq("request_id", request_id) \
        .order("consensus_final_score", desc=True) \
        .execute()

    leads = []
    for row in (consensus_resp.data or []):
        lead_row = {
            "lead_id": row["lead_id"],
            "miner_hotkey": row["miner_hotkey"],
            "consensus_final_score": row["consensus_final_score"],
            "consensus_intent_signal_final": row["consensus_intent_signal_final"],
            "is_winner": row["is_winner"],
            "num_validators": row["num_validators"],
            "any_fabricated": row["any_fabricated"],
            "consensus_tier2_passed": row["consensus_tier2_passed"],
            "consensus_email_verified": row.get("consensus_email_verified"),
            "consensus_person_verified": row.get("consensus_person_verified"),
            "consensus_company_verified": row.get("consensus_company_verified"),
            "consensus_rep_score": row.get("consensus_rep_score"),
        }
        leads.append(lead_row)

    result = {
        "request_id": request_id,
        "request_status": req["status"],
        "leads": leads,
        "total_leads": len(leads),
    }
    if req.get("successor_request_id"):
        result["successor_request_id"] = req["successor_request_id"]
    return result


# ---------------------------------------------------------------
# GET /fulfillment/rewards/active  — validator fetches active rewards
# ---------------------------------------------------------------
@fulfillment_router.get("/rewards/active")
async def get_active_rewards(current_epoch: int):
    """Return active (unexpired) fulfillment rewards grouped by miner hotkey.

    Used by the validator during weight calculation to determine the
    fulfillment emission carve-out from the sourcing allocation.
    """
    supabase = _get_supabase()

    resp = supabase.table("fulfillment_score_consensus") \
        .select("miner_hotkey, reward_pct, reward_expires_epoch") \
        .not_.is_("reward_pct", "null") \
        .gt("reward_expires_epoch", current_epoch) \
        .execute()

    per_miner: dict = {}
    for row in (resp.data or []):
        hk = row["miner_hotkey"]
        pct = float(row["reward_pct"])
        per_miner[hk] = per_miner.get(hk, 0.0) + pct

    return {"rewards": per_miner, "total_active_rows": len(resp.data or [])}


# ---------------------------------------------------------------
# POST /fulfillment/ban/{hotkey}  — validator requests a ban
# ---------------------------------------------------------------
@fulfillment_router.post("/ban/{hotkey}")
async def request_ban(hotkey: str, reason: str = "", validator_hotkey: str = "", request_id: str = ""):
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled")

    _log_event(EventType.FULFILLMENT_BAN, {
        "hotkey": hotkey,
        "reason": reason,
        "banned_by": validator_hotkey or "admin",
        "request_id": request_id,
    })

    if not FULFILLMENT_BANS_ENABLED:
        return {
            "action": "logged_only",
            "detail": "Ban logged but not executed — FULFILLMENT_BANS_ENABLED is false",
        }

    supabase = _get_supabase()
    if request_id:
        check = supabase.table("fulfillment_scores") \
            .select("score_id") \
            .eq("request_id", request_id) \
            .eq("miner_hotkey", hotkey) \
            .eq("all_fabricated", True) \
            .limit(1) \
            .execute()
        if not check.data:
            raise HTTPException(400, detail="No fabricated leads found for this hotkey on the given request")

    success = await ban_hotkey(hotkey, reason or "Fulfillment fabrication", validator_hotkey or "system")
    if not success:
        raise HTTPException(500, detail="Ban execution failed")

    q = supabase.table("fulfillment_score_consensus").update({
        "reward_pct": None,
    }).eq("miner_hotkey", hotkey)
    if request_id:
        q = q.eq("request_id", request_id)
    q.execute()

    return {"action": "banned", "hotkey": hotkey}
