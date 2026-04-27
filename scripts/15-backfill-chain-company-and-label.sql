-- ================================================================
-- Migration 15: backfill missing company / internal_label on
--               descendants of labeled chain roots
-- ================================================================
--
-- Background:
--   The recycle path in gateway/fulfillment/lifecycle.py::_recycle_request
--   propagates ``internal_label`` and ``company`` from predecessor to
--   successor, but that propagation didn't always exist.  Chains rooted
--   before the propagation fix landed have a labeled gen 1 followed by
--   N generations where both fields are NULL.  When such a chain
--   eventually fulfills, the fulfilled row carries no client identity:
--   the ``company`` column on the fulfilled head is NULL, so:
--
--     * Any future create_request for the same client cannot rely on
--       _load_previously_delivered_companies(client) to surface the
--       already-delivered companies — that lookup filters on
--       company=<client> AND status='fulfilled' on the
--       fulfillment_requests table.  A NULL-company fulfilled row
--       won't match either filter, so the 40 just-delivered companies
--       (Atlantic3PL NJ chain root b4006f01 → fulfilled head 5680a076,
--       observed Apr 26 2026) silently fall off the exclusion list and
--       miners can re-deliver them and get paid again.
--     * Any client-attribution query ("how many leads has Atlantic3PL
--       NJ ever received?") returns 0 because every fulfilled row in
--       their chain has company=NULL after gen 1.
--
--   The propagation bug is fixed in current code, so new chains are
--   fine going forward.  This migration is a one-time backfill that
--   walks every existing chain forward from the most upstream row that
--   HAS a populated company/internal_label and copies those values
--   down to all descendants where the corresponding field is currently
--   NULL.  Idempotent (re-running this migration is a no-op).
--
--   It does NOT overwrite existing labels on a descendant — COALESCE
--   keeps the descendant's value if non-NULL.  So if a chain has a
--   mid-chain row that explicitly set a different label (none observed
--   in production but possible in principle), that mid-chain label
--   takes precedence for its own descendants.
-- ================================================================


-- ─────────────────────────────────────────────────────────────────
-- STEP 1 (read-only preview): show every descendant that WOULD be
-- updated, with the labels that will be inherited.  Uncomment to run
-- by hand before applying the UPDATE below.
-- ─────────────────────────────────────────────────────────────────
-- WITH RECURSIVE chain_walk AS (
--     -- Anchor: every row that has a populated company / internal_label
--     -- AND has at least one descendant.  These are our "donor" rows.
--     SELECT
--         request_id,
--         successor_request_id,
--         company,
--         internal_label,
--         request_id      AS donor_id,
--         company         AS donor_company,
--         internal_label  AS donor_label,
--         0               AS depth
--     FROM fulfillment_requests
--     WHERE company IS NOT NULL
--       AND internal_label IS NOT NULL
--       AND successor_request_id IS NOT NULL
--
--     UNION ALL
--
--     -- Recursive: follow successor_request_id forward.  Stop entering a
--     -- row that already has BOTH company and internal_label set
--     -- (because that row is its own donor for its own subchain — a
--     -- separate anchor row will start the walk from there).
--     SELECT
--         fr.request_id,
--         fr.successor_request_id,
--         fr.company,
--         fr.internal_label,
--         cw.donor_id,
--         cw.donor_company,
--         cw.donor_label,
--         cw.depth + 1
--     FROM fulfillment_requests fr
--     INNER JOIN chain_walk cw ON fr.request_id = cw.successor_request_id
--     WHERE (fr.company IS NULL OR fr.internal_label IS NULL)
--       AND cw.depth < 100  -- safety cap; no real chain should exceed this
-- )
-- SELECT
--     cw.request_id,
--     cw.depth,
--     cw.company        AS current_company,
--     cw.internal_label AS current_label,
--     cw.donor_company  AS will_set_company,
--     cw.donor_label    AS will_set_label
-- FROM chain_walk cw
-- WHERE cw.depth > 0
-- ORDER BY cw.donor_id, cw.depth;


-- ─────────────────────────────────────────────────────────────────
-- STEP 2: actual backfill UPDATE.  Idempotent (re-runs are no-ops).
-- ─────────────────────────────────────────────────────────────────
WITH RECURSIVE chain_walk AS (
    SELECT
        request_id,
        successor_request_id,
        company,
        internal_label,
        request_id      AS donor_id,
        company         AS donor_company,
        internal_label  AS donor_label,
        0               AS depth
    FROM fulfillment_requests
    WHERE company IS NOT NULL
      AND internal_label IS NOT NULL
      AND successor_request_id IS NOT NULL

    UNION ALL

    SELECT
        fr.request_id,
        fr.successor_request_id,
        fr.company,
        fr.internal_label,
        cw.donor_id,
        cw.donor_company,
        cw.donor_label,
        cw.depth + 1
    FROM fulfillment_requests fr
    INNER JOIN chain_walk cw ON fr.request_id = cw.successor_request_id
    WHERE (fr.company IS NULL OR fr.internal_label IS NULL)
      AND cw.depth < 100
)
UPDATE fulfillment_requests fr
   SET
       company        = COALESCE(fr.company,        cw.donor_company),
       internal_label = COALESCE(fr.internal_label, cw.donor_label)
  FROM chain_walk cw
 WHERE fr.request_id = cw.request_id
   AND cw.depth > 0
   AND (fr.company IS NULL OR fr.internal_label IS NULL);  -- idempotency guard


-- ─────────────────────────────────────────────────────────────────
-- STEP 3 (verification): count remaining NULL-labeled descendants.
-- After the UPDATE this should return 0 unless there is a chain whose
-- root has NULL labels (in which case the descendants legitimately
-- inherit nothing).
-- ─────────────────────────────────────────────────────────────────
-- WITH RECURSIVE chain_walk AS (
--     SELECT request_id, successor_request_id, 0 AS depth
--     FROM fulfillment_requests
--     WHERE company IS NOT NULL
--       AND internal_label IS NOT NULL
--       AND successor_request_id IS NOT NULL
--     UNION ALL
--     SELECT fr.request_id, fr.successor_request_id, cw.depth + 1
--     FROM fulfillment_requests fr
--     INNER JOIN chain_walk cw ON fr.request_id = cw.successor_request_id
--     WHERE cw.depth < 100
-- )
-- SELECT COUNT(*) AS remaining_null_labeled_descendants
--   FROM chain_walk cw
--   JOIN fulfillment_requests fr ON cw.request_id = fr.request_id
--  WHERE cw.depth > 0
--    AND (fr.company IS NULL OR fr.internal_label IS NULL);


-- ─────────────────────────────────────────────────────────────────
-- Sanity check (optional): expected rows backfilled for the
-- Atlantic3PL chain (root b4006f01-... created Apr 22, fulfilled gen
-- 38 = 5680a076-... on Apr 26).  Run AFTER the UPDATE above to
-- confirm the chain is now consistently labeled.
-- ─────────────────────────────────────────────────────────────────
-- SELECT request_id, status, company, internal_label, num_leads, window_end
--   FROM fulfillment_requests
--  WHERE request_id::text LIKE 'b4006f01%'
--     OR request_id::text LIKE '5680a076%'
--     OR request_id IN (
--         SELECT request_id FROM fulfillment_requests
--          WHERE successor_request_id::text LIKE '5680a076%'
--     )
--  ORDER BY created_at;
