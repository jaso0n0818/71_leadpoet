-- =====================================================================
-- Migration: internal client-facing label on fulfillment_requests
-- =====================================================================
-- Adds a small text column for storing a client identifier such as
-- "Edward Burrowes 1".  Stored OUTSIDE `icp_details` so miners (who only
-- see `icp_details`) never receive it; visible in Supabase dashboards.
--
-- Safe to re-run.
-- =====================================================================

ALTER TABLE public.fulfillment_requests
  ADD COLUMN IF NOT EXISTS internal_label text;

-- Optional index if you'll filter by label in the dashboard.
-- CREATE INDEX IF NOT EXISTS idx_fulfillment_requests_internal_label
--   ON public.fulfillment_requests (internal_label);
