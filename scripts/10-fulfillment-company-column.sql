-- =====================================================================
-- Migration: client company name on fulfillment_requests
-- =====================================================================
-- Mirrors the internal_label pattern.  Required at the API/Pydantic layer,
-- but intentionally nullable on the DB so existing rows (which don't have
-- a company value) stay valid.
--
-- At request-creation time, the gateway ALSO scrubs every whole-word match
-- of this company name from the free-text ICP fields (prompt,
-- product_service, intent_signals, target_roles), replacing each with
-- "[company_name]" so miners never learn which client submitted the
-- request.
--
-- Safe to re-run.
-- =====================================================================

ALTER TABLE public.fulfillment_requests
  ADD COLUMN IF NOT EXISTS company text;

-- Optional index if we ever filter dashboards by client company.
-- CREATE INDEX IF NOT EXISTS idx_fulfillment_requests_company
--   ON public.fulfillment_requests (company);
