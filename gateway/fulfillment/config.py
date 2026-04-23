"""
Fulfillment system configuration.

All values are read from environment variables with sensible defaults.
"""

import os
import logging

T_EPOCHS = int(os.getenv("FULFILLMENT_T_EPOCHS", "2"))
T_SECONDS_OVERRIDE = int(os.getenv("FULFILLMENT_T_SECONDS", "0"))
M_MINUTES = int(os.getenv("FULFILLMENT_M_MINUTES", "15"))
BLOCK_TIME_SECONDS = 12
Z_PERCENT = float(os.getenv("FULFILLMENT_Z_PERCENT", "0.001"))
L_EPOCHS = int(os.getenv("FULFILLMENT_L_EPOCHS", "30"))
FULFILLMENT_MAX_CONCURRENT_SOURCES = int(os.getenv("FULFILLMENT_MAX_CONCURRENT_SOURCES", "2"))
FULFILLMENT_OPENROUTER_API_KEY = os.getenv("FULFILLMENT_OPENROUTER_API_KEY", "")
FULFILLMENT_LIFECYCLE_INTERVAL_SECONDS = int(os.getenv("FULFILLMENT_LIFECYCLE_INTERVAL_SECONDS", "30"))
FULFILLMENT_MIN_VALIDATORS = int(os.getenv("FULFILLMENT_MIN_VALIDATORS", "1"))
# How long the gateway waits (after reveal_window_end) for validators to
# score a request before recycling it with reason=no_validators_timeout.
# Must exceed the validator's worst-case end-to-end scoring time, not
# just its polling cadence.  Scoring a single request with 60+ leads
# takes 30-60 minutes under real load (Stage 4 LinkedIn scrapes, Tier
# 3 LLM calls per signal, TrueList batch verification, per-lead rep
# score lookup).  5 min was too tight (2026-04-21).  15 min was also
# too tight (2026-04-22: requests 669fd2b7 and 7c666b4e both expired
# with no consensus computed even though the validator DID score them
# minutes later — scores landed in fulfillment_scores after the gateway
# had already marked the request expired, orphaning real miner work).
# 90 min gives the validator plenty of headroom for a backed-up scoring
# queue across multiple concurrent requests and is still well under the
# 72-min epoch × multiple-epoch runway that the reward system allows.
FULFILLMENT_CONSENSUS_TIMEOUT_MINUTES = int(os.getenv("FULFILLMENT_CONSENSUS_TIMEOUT_MINUTES", "90"))
FULFILLMENT_BANS_ENABLED = os.getenv("FULFILLMENT_BANS_ENABLED", "false").lower() == "true"

FULFILLMENT_MAX_PARALLEL_REQUESTS = int(os.getenv("FULFILLMENT_MAX_PARALLEL_REQUESTS", "5"))
FULFILLMENT_MIN_REMAINING_WINDOW_MINUTES = int(os.getenv("FULFILLMENT_MIN_REMAINING_WINDOW_MINUTES", "15"))

# Per-miner submission cap is (request.num_leads * this multiplier), ceil'd.
# Default 1.5 so a miner can commit ~50% more leads than the request requires
# to absorb the real-time validation flakiness (TrueList + LinkedIn scrapes
# currently pass ~70-80% of legitimate leads).  Only the top num_leads by
# score actually win rewards — the surplus just protects the miner from
# having their whole batch discarded because a couple of leads lost the
# coin flip on a transient failure.
#
# Increase on days with low miner participation, decrease once pass-rate
# improves.  Must be >= 1.0 (can't commit fewer than num_leads or the
# quota gate can never be met from a single miner).
FULFILLMENT_MINER_SUBMISSION_MULTIPLIER = float(os.getenv("FULFILLMENT_MINER_SUBMISSION_MULTIPLIER", "1.5"))
if FULFILLMENT_MINER_SUBMISSION_MULTIPLIER < 1.0:
    logging.warning(
        f"FULFILLMENT_MINER_SUBMISSION_MULTIPLIER={FULFILLMENT_MINER_SUBMISSION_MULTIPLIER} < 1.0; "
        "clamping to 1.0 so miners can always commit at least num_leads."
    )
    FULFILLMENT_MINER_SUBMISSION_MULTIPLIER = 1.0

FULFILLMENT_MIN_INTENT_SCORE = float(os.getenv("FULFILLMENT_MIN_INTENT_SCORE", "5.0"))
FULFILLMENT_INTENT_QUALITY_FLOOR = float(os.getenv("FULFILLMENT_INTENT_QUALITY_FLOOR", "5.0"))
FULFILLMENT_INTENT_BREADTH_WEIGHT = float(os.getenv("FULFILLMENT_INTENT_BREADTH_WEIGHT", "0.10"))

if os.getenv("ENABLE_FULFILLMENT", "false").lower() == "true" and not FULFILLMENT_OPENROUTER_API_KEY:
    logging.warning(
        "ENABLE_FULFILLMENT=true but FULFILLMENT_OPENROUTER_API_KEY is not set. "
        "Fulfillment scoring will fail when the first request is processed. "
        "Set the env var or disable fulfillment with ENABLE_FULFILLMENT=false."
    )


def epochs_to_seconds(num_epochs: int, tempo: int = 360) -> int:
    """Convert an epoch count to wall-clock seconds using the subnet tempo."""
    return num_epochs * tempo * BLOCK_TIME_SECONDS


def get_fulfillment_api_key() -> str:
    """Lazy validation — only raises when fulfillment scoring is actually invoked."""
    if not FULFILLMENT_OPENROUTER_API_KEY:
        raise ValueError(
            "FULFILLMENT_OPENROUTER_API_KEY must be set when fulfillment scoring is active. "
            "Set the env var or disable fulfillment with ENABLE_FULFILLMENT=false."
        )
    return FULFILLMENT_OPENROUTER_API_KEY
