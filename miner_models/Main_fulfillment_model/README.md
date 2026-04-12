# Fulfillment Lead Model

Discovers companies matching an ICP from the web, finds decision-maker contacts,
verifies emails via TrueList, and enriches with intent signals using Perplexity + ScrapingDog.

## Setup

```bash
pip install httpx fastapi pydantic python-dotenv
export SCRAPINGDOG_API_KEY=your_key
export OPENROUTER_KEY=your_key
export TRUELIST_API_KEY=your_key
```

## Usage

```python
from target_fit_model.discovery import source_fulfillment_leads

leads = await source_fulfillment_leads(
    icp={
        "industry": "Software",
        "sub_industry": "SaaS",
        "target_roles": ["VP of Sales"],
        "intent_signals": ["hiring SDRs", "evaluating sales tools"],
        "country": "United States",
    },
    num_leads=5,
)
```
