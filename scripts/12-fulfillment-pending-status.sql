-- ================================================================
-- Migration 12: Add 'pending' status for queue-before-visible workflow
-- ================================================================
--
-- Background:
--   Before this migration a fulfillment request was created in 'open'
--   status with its commit window (window_start, window_end) already
--   ticking.  The miner-facing /fulfillment/requests/active endpoint
--   returned up to FULFILLMENT_MAX_PARALLEL_REQUESTS (5) rows ordered
--   by window_start.  When more than 5 open requests existed, the
--   6th+ tickled down their commit timers while invisible to miners,
--   and could expire before ever being seen.
--
-- New workflow:
--   * Request is inserted in status='pending' with NULL window
--     timestamps.  No commit timer is running.
--   * The lifecycle tick promotes the oldest 'pending' rows to 'open'
--     (and stamps window_start=NOW, window_end=NOW + T_EPOCHS,
--     reveal_window_end=window_end + M_MINUTES) whenever the count of
--     'open' rows drops below FULFILLMENT_MAX_PARALLEL_REQUESTS.
--   * Miners therefore only ever see requests whose commit window
--     started when they became visible.
--
-- Action:
--   1. Add 'pending' to the status CHECK constraint.
--   2. Drop NOT NULL on window_start / window_end / reveal_window_end
--      so pending rows can have NULL timers.
-- ================================================================

ALTER TABLE fulfillment_requests
    DROP CONSTRAINT IF EXISTS fulfillment_requests_status_check;

ALTER TABLE fulfillment_requests
    ADD CONSTRAINT fulfillment_requests_status_check
    CHECK (status IN (
        'pending',
        'open',
        'commit_closed',
        'scoring',
        'fulfilled',
        'recycled',
        'expired'
    ));

ALTER TABLE fulfillment_requests ALTER COLUMN window_start DROP NOT NULL;
ALTER TABLE fulfillment_requests ALTER COLUMN window_end DROP NOT NULL;
ALTER TABLE fulfillment_requests ALTER COLUMN reveal_window_end DROP NOT NULL;

-- Verify:
-- SELECT conname, pg_get_constraintdef(oid)
--   FROM pg_constraint
--  WHERE conrelid = 'fulfillment_requests'::regclass
--    AND contype = 'c';
