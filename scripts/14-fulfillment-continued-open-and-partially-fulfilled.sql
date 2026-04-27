-- ================================================================
-- Migration 14: Add 'continued_open' and 'partially_fulfilled' statuses
-- ================================================================
--
-- Background:
--   Before this migration the fulfillment lifecycle had two failure
--   modes when a request didn't reach its full client-requested
--   num_leads quota in one cycle:
--     * status='recycled': always — regardless of whether 0 or N-1 leads
--       cleared validation.  All validated leads were silently dropped
--       on the floor and a fresh successor was created.
--   This wasted Apify $$$ on every miner submission whose batch fell
--   short and gave miners no signal that their work was real but
--   awaiting a multi-cycle fulfillment.
--
-- New behavior:
--   * 'partially_fulfilled' — a former 'recycled' request that DID
--     produce ≥1 chain-held lead (is_chain_held=TRUE in
--     fulfillment_score_consensus).  The successor request inherits
--     those held leads' companies as additional excluded_companies and
--     asks miners only for the REMAINING quota
--     (chain_target − held_count).  Held leads do NOT yet earn
--     rewards — they pay out only when the chain reaches 'fulfilled'.
--
--   * 'continued_open' — successor of a 'partially_fulfilled' request.
--     Behaves identically to 'open' for commit/reveal windowing and
--     miner visibility (/fulfillment/requests/active surfaces both),
--     but the dedicated label tells miners "this is a continuation;
--     prior held leads exist; you won't earn rewards until the chain
--     reaches its full quota across all generations".
--
--   * 'recycled' is reserved for the no-progress case (0 chain-held
--     leads after consensus — empty reveals, empty consensus, all
--     leads failed every gate).  Successors of 'recycled' are plain
--     'open' requests starting fresh, with no in-flight held set.
--
-- Status flow (full picture):
--
--    pending ─→ open / continued_open ─→ commit_closed ─→ scoring ─→ fulfilled
--                                                                ├─→ partially_fulfilled  (chain continues)
--                                                                ├─→ recycled             (chain restarts)
--                                                                └─→ expired              (no validators / unrecoverable)
--
-- Action:
--   1. Drop the old status check constraint.
--   2. Re-add it with the two new statuses appended.
--
-- Backward compatibility:
--   Existing rows with status='recycled' or 'fulfilled' or any other
--   pre-existing value are unaffected.  No backfill needed.
-- ================================================================

ALTER TABLE fulfillment_requests
    DROP CONSTRAINT IF EXISTS fulfillment_requests_status_check;

ALTER TABLE fulfillment_requests
    ADD CONSTRAINT fulfillment_requests_status_check
    CHECK (status IN (
        'pending',
        'open',
        'continued_open',
        'commit_closed',
        'scoring',
        'fulfilled',
        'partially_fulfilled',
        'recycled',
        'expired'
    ));

-- Verify:
-- SELECT conname, pg_get_constraintdef(oid)
--   FROM pg_constraint
--  WHERE conrelid = 'fulfillment_requests'::regclass
--    AND contype = 'c';
