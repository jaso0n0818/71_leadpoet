-- ================================================================
-- Migration 11: Allow status='expired' on fulfillment_requests
-- ================================================================
--
-- Background:
--   gateway/fulfillment/lifecycle.py sets status='expired' via
--   _recycle_request(..., terminal_status="expired") when a scoring
--   request times out with zero validators or produces empty consensus.
--   The existing CHECK constraint only allowed
--   ('open','commit_closed','scoring','fulfilled','recycled'), so every
--   such recycle attempt raised
--     "new row for relation \"fulfillment_requests\" violates check
--      constraint \"fulfillment_requests_status_check\""
--   and the request was silently wedged in 'scoring' forever.
--
--   Observed pre-migration state (2026-04-19):
--     status='scoring': 9 rows (all stuck — their recycle attempts
--     have been failing every 30s for hours).
--
-- Action:
--   1. Replace the CHECK constraint so 'expired' is a legal value.
--   2. The 9 stuck 'scoring' rows will be recycled automatically on
--      the next lifecycle tick once this migration is applied.  No
--      manual backfill required.
-- ================================================================

ALTER TABLE fulfillment_requests
    DROP CONSTRAINT IF EXISTS fulfillment_requests_status_check;

ALTER TABLE fulfillment_requests
    ADD CONSTRAINT fulfillment_requests_status_check
    CHECK (status IN (
        'open',
        'commit_closed',
        'scoring',
        'fulfilled',
        'recycled',
        'expired'
    ));

-- Verify the new constraint is in place:
-- SELECT conname, pg_get_constraintdef(oid)
--   FROM pg_constraint
--  WHERE conrelid = 'fulfillment_requests'::regclass
--    AND contype = 'c';
