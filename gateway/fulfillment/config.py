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
FULFILLMENT_CONSENSUS_TIMEOUT_MINUTES = int(os.getenv("FULFILLMENT_CONSENSUS_TIMEOUT_MINUTES", "5"))
FULFILLMENT_BANS_ENABLED = os.getenv("FULFILLMENT_BANS_ENABLED", "false").lower() == "true"

FULFILLMENT_MAX_PARALLEL_REQUESTS = int(os.getenv("FULFILLMENT_MAX_PARALLEL_REQUESTS", "5"))

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
