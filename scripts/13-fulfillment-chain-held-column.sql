-- Migration 13: add is_chain_held to fulfillment_score_consensus
--
-- WHY:
--   The fulfillment lifecycle now supports partial-quota chains.  Earlier
--   the only outcome of a request that produced fewer winners than its
--   client-requested num_leads was "discard everything and recycle".
--   That meant Apify-validated leads (with real $ cost spent) were
--   dropped on the floor every time miners couldn't deliver a full
--   batch in one cycle.
--
--   With the new state machine a request whose consensus produced
--   ANY winners transitions to status='partially_fulfilled' instead
--   of 'recycled'.  The validated winners are HELD across recycle
--   generations: their consensus rows stay in the table with
--   is_chain_held=TRUE.  Each successor request asks miners only for
--   the REMAINING quota (chain_target − held_count) and adds the
--   currently-held companies to its excluded_companies list so miners
--   don't re-do work we've already accepted.
--
--   When a successor cycle produces NEW candidates that out-score the
--   currently-held set, the displaced held leads flip is_chain_held
--   back to FALSE.  Their companies drop off the next exclusion list
--   automatically.  Only when len(held) ≥ chain_target does the
--   request transition to 'fulfilled' and rewards flow (is_winner
--   gets set, reward_pct populated).  Until then no miner gets paid.
--
-- COLUMN SEMANTICS:
--   is_chain_held=TRUE   →  this lead is currently in the chain's
--                           top-K held set (transient, can flip).
--   is_chain_held=FALSE  →  not currently held (either never made it,
--                           or was displaced by a higher-scoring entry
--                           in a later generation).
--
--   Independent of is_winner:
--   * is_winner=TRUE    →  paid rewards; only set on chain fulfillment
--                          (immutable, ties to reward_pct).
--   * is_chain_held=TRUE →  candidate for is_winner if/when chain
--                          fulfills; transient until then.
--
-- BACKWARD COMPAT:
--   Existing rows pre-migration get is_chain_held=FALSE (the column
--   default).  Old completed chains either had is_winner=TRUE (paid
--   rewards on fulfillment) or were silently zeroed on recycle —
--   neither path needs is_chain_held set, so default FALSE is
--   correct for all historical data.

ALTER TABLE fulfillment_score_consensus
  ADD COLUMN IF NOT EXISTS is_chain_held BOOLEAN NOT NULL DEFAULT FALSE;

-- Index helps the chain-walking query in
-- gateway/fulfillment/lifecycle.py::_load_chain_held_winners which
-- frequently filters by (request_id, is_chain_held=TRUE).
CREATE INDEX IF NOT EXISTS idx_consensus_chain_held
  ON fulfillment_score_consensus (request_id)
  WHERE is_chain_held = TRUE;
