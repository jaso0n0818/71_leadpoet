"""
FastAPI endpoint for the Fulfillment Lead Model.

Provides REST API access to the lead discovery + intent enrichment pipeline.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/leads", tags=["Fulfillment Leads"])


class DiscoverRequest(BaseModel):
    industry: str = Field(..., description="Required. e.g., 'Software'")
    sub_industry: Optional[str] = Field(None, description="e.g., 'SaaS'")
    target_roles: Optional[List[str]] = Field(None, description="e.g., ['VP of Sales']")
    target_seniority: Optional[str] = Field(None, description="e.g., 'VP'")
    employee_count: Optional[str] = Field(None, description="e.g., '50-500'")
    country: Optional[str] = Field(None, description="e.g., 'United States'")
    product_service: Optional[str] = Field(None, description="Product/service description")
    intent_signals: Optional[List[str]] = Field(None, description="List of intent signals to search for")
    prompt: Optional[str] = Field(None, description="Natural language ICP description")
    num_leads: int = Field(5, ge=1, le=50, description="Number of leads to return")


@router.post("/discover")
async def discover_leads(request: DiscoverRequest):
    """Discover leads matching an ICP with verified intent signals."""
    from target_fit_model.discovery import source_fulfillment_leads

    icp = request.model_dump()
    result = await source_fulfillment_leads(icp, num_leads=request.num_leads)
    return {"leads": result, "count": len(result)}
