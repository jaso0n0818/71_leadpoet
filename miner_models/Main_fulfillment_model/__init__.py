"""
Fulfillment Lead Model

Discovers companies matching an ICP from the web, finds decision-maker
contacts, verifies emails via TrueList, and enriches with intent signals
using Perplexity + ScrapingDog.
"""

import importlib
import os
import sys

# Register this package as 'target_fit_model' so that internal
# cross-module imports (e.g. intent_enrichment → config, openrouter)
# resolve correctly regardless of how the package is invoked.
_pkg_dir = os.path.dirname(__file__)
if "target_fit_model" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "target_fit_model",
        os.path.join(_pkg_dir, "__init__.py"),
        submodule_search_locations=[_pkg_dir],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["target_fit_model"] = mod
    # Don't exec_module here — just register the path so submodule
    # imports like target_fit_model.config resolve via the search locations.

__all__ = ["source_fulfillment_leads"]
