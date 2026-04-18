"""
Fulfillment lifecycle background task.

Manages request state transitions, consensus aggregation, reward expiry,
and request recycling.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from dateutil.parser import isoparse as _isoparse

from gateway.fulfillment.config import (
    FULFILLMENT_LIFECYCLE_INTERVAL_SECONDS,
    FULFILLMENT_MIN_VALIDATORS,
    FULFILLMENT_CONSENSUS_TIMEOUT_MINUTES,
    L_EPOCHS,
    Z_PERCENT,
)
from gateway.fulfillment.consensus import compute_fulfillment_consensus
from gateway.models.events import EventType

logger = logging.getLogger(__name__)


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


_ADVISORY_LOCK_KEY = int.from_bytes(
    __import__("hashlib").sha256(b"fulfillment_lifecycle").digest()[:4],
    "big",
) % (2**31)


def _try_advisory_lock(supabase) -> bool:
    """Acquire PostgreSQL advisory lock to prevent duplicate processing
    across multiple gateway instances.  Calls a public-schema wrapper
    because pg_catalog functions are not exposed via PostgREST."""
    try:
        resp = supabase.rpc("fulfillment_try_lifecycle_lock", {
            "p_key": _ADVISORY_LOCK_KEY,
        }).execute()
        return bool(resp.data)
    except Exception as e:
        logger.warning(f"Advisory lock acquire failed (proceeding without lock): {e}")
        return True


def _release_advisory_lock(supabase) -> None:
    """Release the advisory lock after processing."""
    try:
        supabase.rpc("fulfillment_release_lifecycle_lock", {
            "p_key": _ADVISORY_LOCK_KEY,
        }).execute()
    except Exception as e:
        logger.warning(f"Advisory lock release failed: {e}")


async def fulfillment_lifecycle_task() -> None:
    """Background loop managing fulfillment request state transitions."""
    print("🔄 Fulfillment lifecycle task running (every 30s)")

    while True:
        try:
            await _lifecycle_tick()
        except asyncio.CancelledError:
            print("Fulfillment lifecycle task cancelled")
            break
        except Exception as e:
            print(f"❌ Fulfillment lifecycle error: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(FULFILLMENT_LIFECYCLE_INTERVAL_SECONDS)


async def _lifecycle_tick() -> None:
    supabase = _get_supabase()

    if not _try_advisory_lock(supabase):
        logger.debug("Lifecycle tick skipped — another instance holds the lock")
        return

    try:
        await _lifecycle_tick_inner(supabase)
    finally:
        _release_advisory_lock(supabase)


async def _lifecycle_tick_inner(supabase) -> None:
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    # Debug: show all non-terminal request statuses
    all_req = supabase.table("fulfillment_requests") \
        .select("request_id, status, window_end, reveal_window_end") \
        .in_("status", ["open", "commit_closed", "scoring"]) \
        .execute()
    if all_req.data:
        print(f"🔄 Lifecycle tick @ {now_iso[:19]}Z — {len(all_req.data)} active request(s):")
        for ar in (all_req.data or []):
            print(f"   {ar['request_id'][:8]}... status={ar['status']} "
                  f"window_end={ar.get('window_end', '?')[:19]} "
                  f"reveal_end={ar.get('reveal_window_end', '?')[:19]}")

    # Step 1: open -> commit_closed (past window_end)
    open_past_window = supabase.table("fulfillment_requests") \
        .select("request_id") \
        .eq("status", "open") \
        .lt("window_end", now_iso) \
        .execute()
    if open_past_window.data:
        print(f"Lifecycle: {len(open_past_window.data)} open request(s) past window_end")
    for r in (open_past_window.data or []):
        try:
            supabase.rpc("fulfillment_close_window", {
                "p_request_id": r["request_id"],
                "p_new_status": "commit_closed",
            }).execute()
            # Verify the transition actually happened
            verify = supabase.table("fulfillment_requests") \
                .select("status") \
                .eq("request_id", r["request_id"]) \
                .execute()
            actual_status = verify.data[0]["status"] if verify.data else "?"
            print(f"   {r['request_id'][:8]}... -> commit_closed (verified: {actual_status})")
        except Exception as e:
            print(f"   Error closing {r['request_id'][:8]}...: {e}")

    # Step 2: commit_closed -> scoring or recycled (past reveal_window_end)
    closed_past_reveal = supabase.table("fulfillment_requests") \
        .select("request_id, icp_details, num_leads, reveal_window_end") \
        .eq("status", "commit_closed") \
        .lt("reveal_window_end", now_iso) \
        .execute()

    if closed_past_reveal.data:
        print(f"Lifecycle Step 2: {len(closed_past_reveal.data)} commit_closed request(s) past reveal_window_end")

    for r in (closed_past_reveal.data or []):
        rid = r["request_id"]
        print(f"   Checking {rid[:8]}... (reveal_window_end={r.get('reveal_window_end', '?')})")

        all_subs = supabase.table("fulfillment_submissions") \
            .select("submission_id, revealed, miner_hotkey") \
            .eq("request_id", rid) \
            .execute()
        print(f"   Total submissions for {rid[:8]}: {len(all_subs.data or [])}")
        for s in (all_subs.data or []):
            print(f"     sub={s['submission_id'][:8]}... miner={s['miner_hotkey'][:8]}... revealed={s['revealed']}")

        reveals = supabase.table("fulfillment_submissions") \
            .select("submission_id") \
            .eq("request_id", rid) \
            .eq("revealed", True) \
            .execute()
        print(f"   Revealed submissions: {len(reveals.data or [])}")

        if reveals.data:
            try:
                supabase.rpc("fulfillment_close_window", {
                    "p_request_id": rid,
                    "p_new_status": "scoring",
                }).execute()
                print(f"   ✅ {rid[:8]}... -> scoring ({len(reveals.data)} reveal(s))")
            except Exception as e:
                print(f"   ❌ Error transitioning {rid[:8]}... to scoring: {e}")
        else:
            print(f"   ⚠️  No reveals for {rid[:8]}... — recycling")
            _recycle_request(
                supabase, r, now, now_iso,
                terminal_status="recycled",
                reason="no_reveals",
            )

    # Step 3: consensus aggregation for scoring requests
    scoring_requests = supabase.table("fulfillment_requests") \
        .select("request_id, reveal_window_end, icp_details, num_leads") \
        .eq("status", "scoring") \
        .execute()

    if scoring_requests.data:
        print(f"Lifecycle Step 3: {len(scoring_requests.data)} request(s) in scoring status")

    for r in (scoring_requests.data or []):
        rid = r["request_id"]
        try:
            validator_count_resp = supabase.table("fulfillment_scores") \
                .select("validator_hotkey") \
                .eq("request_id", rid) \
                .execute()
            unique_validators = {s["validator_hotkey"] for s in (validator_count_resp.data or [])}

            reveal_end = _isoparse(r["reveal_window_end"])
            timeout = reveal_end + timedelta(minutes=FULFILLMENT_CONSENSUS_TIMEOUT_MINUTES)

            if len(unique_validators) < FULFILLMENT_MIN_VALIDATORS and now < timeout:
                mins_left = (timeout - now).total_seconds() / 60
                print(f"   {rid[:8]}... waiting for validators: {len(unique_validators)}/{FULFILLMENT_MIN_VALIDATORS} ({mins_left:.1f}min until timeout)")
                continue

            if len(unique_validators) == 0 and now >= timeout:
                print(
                    f"   ⚠️  {rid[:8]}... has 0 validators after timeout — "
                    f"expiring and recycling"
                )
                _recycle_request(
                    supabase, r, now, now_iso,
                    terminal_status="expired",
                    reason="no_validators_timeout",
                )
                continue

            if len(unique_validators) < FULFILLMENT_MIN_VALIDATORS:
                print(
                    f"   ⚠️  {rid[:8]}... consensus timeout: "
                    f"{len(unique_validators)}/{FULFILLMENT_MIN_VALIDATORS} validators — proceeding"
                )

            consensus_results = await compute_fulfillment_consensus(rid)
            if not consensus_results:
                print(
                    f"   ⚠️  {rid[:8]}... produced empty consensus — "
                    f"expiring and recycling"
                )
                _recycle_request(
                    supabase, r, now, now_iso,
                    terminal_status="expired",
                    reason="empty_consensus",
                )
                continue

            supabase.rpc("fulfillment_upsert_consensus", {
                "p_consensus": consensus_results,
            }).execute()

            num_requested = r.get("num_leads") or (r.get("icp_details", {}) or {}).get("num_leads") or 0
            winner_lead_ids = await _run_dedup_and_rewards(rid, consensus_results, num_requested)

            # Quota gate: only mark `fulfilled` when the client's full N-lead
            # quota is satisfied. Otherwise discard this batch and recycle
            # into a fresh successor so miners can take another pass at the
            # same ICP. Clients never see a partial-fulfillment state.
            if num_requested > 0 and len(winner_lead_ids) < num_requested:
                print(
                    f"   ♻️  {rid[:8]}... insufficient winners "
                    f"({len(winner_lead_ids)}/{num_requested}) — recycling"
                )
                _recycle_request(
                    supabase, r, now, now_iso,
                    terminal_status="recycled",
                    reason=f"insufficient_winners_{len(winner_lead_ids)}_of_{num_requested}",
                )
                continue

            supabase.table("fulfillment_requests").update({
                "status": "fulfilled",
            }).eq("request_id", rid).execute()
            print(f"   ✅ {rid[:8]}... -> fulfilled ({len(winner_lead_ids)}/{num_requested} winners from {len(consensus_results)} scored)")

            ranked = sorted(consensus_results, key=lambda x: x.get("consensus_final_score", 0), reverse=True)
            print(f"\n{'='*60}")
            print(f"🏆 FULFILLMENT RESULTS — Request {rid[:8]}...")
            print(f"   {len(ranked)} leads scored, client requested {num_requested}")
            print(f"{'='*60}")
            for i, cr in enumerate(ranked, 1):
                miner = cr.get("miner_hotkey", "?")[:16]
                score = cr.get("consensus_final_score", 0)
                t2 = "✅" if cr.get("consensus_tier2_passed") else "❌"
                is_winner = cr.get("lead_id") in winner_lead_ids
                winner = "👑" if is_winner else "  "
                lid = cr.get("lead_id", "?")[:8]
                print(f"   {winner} #{i}: score={score:.1f} tier2={t2} miner={miner}... lead={lid}...")
            print(f"\n   Winners: {len(winner_lead_ids)}/{len(ranked)} leads")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"   ❌ Error in consensus for {rid[:8]}...: {e}")
            import traceback
            traceback.print_exc()

    # Step 4: reward expiry
    try:
        _expire_rewards(supabase)
    except Exception as e:
        print(f"❌ Reward expiry error: {e}")


async def _run_dedup_and_rewards(request_id: str, consensus_results: list, num_leads: int = 0) -> set:
    """Deduplicate across miners and assign rewards. Returns set of winner lead_ids.

    Quota gate: a request is only considered fulfillable when the number of
    unique deduped leads with ``score > 0`` is GREATER THAN OR EQUAL TO
    ``num_leads``.  If the quota is not met, this function returns an empty
    set WITHOUT writing any rewards — the caller is expected to recycle the
    request into a fresh successor. Clients only receive a complete batch
    of N leads; partial fulfillment is not emitted and miners who submitted
    an insufficient batch get no reward for this request (they can
    re-participate in the successor).

    If ``num_leads`` is 0 (legacy / unspecified), all deduped leads are
    rewarded as before.
    """
    from gateway.fulfillment.rewards import calculate_lead_rewards
    supabase = _get_supabase()

    needed_subs = {r["submission_id"] for r in consensus_results
                   if r["consensus_final_score"] > 0}
    sub_lead_data: dict = {}
    if needed_subs:
        sub_resp = supabase.table("fulfillment_submissions") \
            .select("submission_id, lead_data") \
            .in_("submission_id", list(needed_subs)) \
            .execute()
        for row in (sub_resp.data or []):
            ld_list = row.get("lead_data") or []
            lookup = {ld.get("lead_id"): ld.get("data", {}) for ld in ld_list}
            sub_lead_data[row["submission_id"]] = lookup

    groups: dict = {}
    for r in consensus_results:
        if r["consensus_final_score"] <= 0:
            continue

        lead_info = sub_lead_data.get(r["submission_id"], {}).get(r["lead_id"])
        if not lead_info:
            continue

        # Cross-miner dedup: one lead per company in the final ranking.
        # Different contacts at the same company from different miners collapse
        # into one group; the highest-scoring contact wins and the rest drop.
        company = _normalize_company(lead_info.get("business", ""))
        if not company:
            continue
        dedup_key = company

        if dedup_key not in groups:
            groups[dedup_key] = []
        groups[dedup_key].append(r)

    # Pick the best candidate(s) per dedup group (tied miners on same lead)
    group_results = []  # list of (best_score, best_raw, tied_candidates)
    for dedup_key, candidates in groups.items():
        candidates.sort(key=lambda x: (
            -x["consensus_final_score"],
            -x.get("consensus_intent_signal_final", 0),
        ))

        best_score = candidates[0]["consensus_final_score"]
        best_raw = candidates[0].get("consensus_intent_signal_final", 0)
        tied = [c for c in candidates
                if c["consensus_final_score"] == best_score
                and c.get("consensus_intent_signal_final", 0) == best_raw]
        group_results.append((best_score, best_raw, tied))

    # Rank unique leads (dedup groups) by score
    group_results.sort(key=lambda g: (-g[0], -g[1]))

    # Quota gate: if the client asked for N leads, require at least N unique
    # qualifying deduped leads. Otherwise return empty (no rewards written);
    # caller will recycle into a fresh successor.
    if num_leads > 0 and len(group_results) < num_leads:
        print(
            f"   ⚠️  {request_id[:8]}... quota not met: "
            f"{len(group_results)} unique leads with score>0 < {num_leads} requested — "
            f"no rewards assigned, caller should recycle"
        )
        return set()

    top_groups = group_results[:num_leads] if num_leads > 0 else group_results

    # Flatten to winners; tie_count reflects only tied miners SELECTED (not dropped)
    winners = []
    for _, _, tied in top_groups:
        for c in tied:
            winners.append({**c, "tie_count": len(tied)})

    current_epoch = _get_current_epoch()
    calculate_lead_rewards(request_id, winners, Z_PERCENT, current_epoch, L_EPOCHS)

    return {w["lead_id"] for w in winners}


def _recycle_request(
    supabase,
    original_request: dict,
    now: datetime,
    now_iso: str,
    *,
    terminal_status: str,
    reason: str,
) -> None:
    """Create a successor request and mark the original as terminal.

    Used when a request can't complete normally (no reveals, no validators,
    empty consensus). The successor is a fresh ``open`` request with new
    commit/reveal windows, added to the BACK of the FIFO queue via a new
    ``window_start = now``.  The original is marked ``recycled`` (if nobody
    responded) or ``expired`` (if validators failed to score) so dashboards
    can distinguish why a request was recycled.
    """
    from gateway.fulfillment.config import T_EPOCHS, M_MINUTES, epochs_to_seconds
    rid = original_request["request_id"]
    new_id = str(uuid4())
    tempo = _get_tempo(supabase)
    new_window_end = now + timedelta(seconds=epochs_to_seconds(T_EPOCHS, tempo))
    new_reveal_end = new_window_end + timedelta(minutes=M_MINUTES)

    try:
        # Claim the predecessor FIRST with a guarded UPDATE: only proceed if the
        # predecessor doesn't already have a successor. This is idempotent and
        # race-free — if two lifecycle ticks try to recycle the same request,
        # only one succeeds in claiming it; the other sees zero affected rows
        # and exits without creating an orphan successor.
        #
        # The previous implementation did INSERT-then-UPDATE without a guard,
        # which under concurrent ticks produced orphan successor rows (both
        # ticks would INSERT, only the last UPDATE would win, leaving the
        # first successor with no predecessor pointing to it).
        claim = supabase.table("fulfillment_requests").update({
            "status": terminal_status,
            "successor_request_id": new_id,
        }).eq("request_id", rid) \
          .is_("successor_request_id", "null") \
          .execute()

        if not claim.data:
            # Another tick already recycled this request — do nothing.
            return

        # We own the backlink; safe to insert the successor.
        supabase.table("fulfillment_requests").insert({
            "request_id": new_id,
            "request_hash": "",
            "icp_details": original_request["icp_details"],
            "num_leads": original_request["num_leads"],
            "window_start": now_iso,
            "window_end": new_window_end.isoformat(),
            "reveal_window_end": new_reveal_end.isoformat(),
            "status": "open",
            "created_by": "recycled",
        }).execute()

        _log_event(EventType.FULFILLMENT_RECYCLED, {
            "old_request_id": rid,
            "new_request_id": new_id,
            "reason": reason,
            "terminal_status": terminal_status,
        })
        print(f"   ♻️  {rid[:8]}... {terminal_status} -> {new_id[:8]}... (reason={reason})")
    except Exception as e:
        print(f"   ❌ Error recycling {rid[:8]}...: {e}")


def _normalize_company(name: str) -> str:
    """Normalize company name for dedup."""
    import re
    name = name.lower().strip()
    suffixes = (
        r"\b(inc\.?|llc|ltd\.?|corp\.?|corporation|co\.?|company|"
        r"plc|gmbh|ag|sa|sas|srl|bv|nv|pty|pvt)\b"
    )
    name = re.sub(suffixes, "", name, flags=re.IGNORECASE)
    name = re.sub(r"[,.\s]+$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _get_current_epoch() -> int:
    """Return the current Bittensor epoch ID.

    Uses the gateway's canonical epoch helper, which derives the epoch from
    the live chain block. The previous implementation queried a
    ``subnet_state`` table that does not exist in the Supabase schema, so
    it always silently returned 0 — which made every freshly-awarded
    ``reward_expires_epoch`` equal to ``L_EPOCHS`` (e.g. 30) instead of
    ``current_epoch + L_EPOCHS`` (e.g. 22227). That caused every winning
    fulfillment lead to be treated as already-expired and never earn
    emission.
    """
    try:
        from gateway.utils.epoch import get_current_epoch_id
        return int(get_current_epoch_id())
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            f"_get_current_epoch() fell back to 0: {e}"
        )
        return 0


def _expire_rewards(supabase) -> None:
    """NULL out reward_pct on expired consensus rows."""
    current_epoch = _get_current_epoch()
    if current_epoch <= 0:
        return
    try:
        supabase.table("fulfillment_score_consensus").update({
            "reward_pct": None,
        }).lte("reward_expires_epoch", current_epoch).not_is("reward_pct", None).execute()
    except Exception:
        try:
            resp = supabase.table("fulfillment_score_consensus") \
                .select("consensus_id, reward_pct") \
                .lte("reward_expires_epoch", current_epoch) \
                .execute()
            for row in (resp.data or []):
                if row.get("reward_pct") is not None:
                    supabase.table("fulfillment_score_consensus").update({
                        "reward_pct": None,
                    }).eq("consensus_id", row["consensus_id"]).execute()
        except Exception as e:
            logger.error(f"Reward expiry fallback failed: {e}")
