from fastapi import APIRouter, Query
import datetime

from typing import Optional, List
from models import TestEndpointResponse

router = APIRouter(tags=["test"])

# 確認テスト諸々(後で実装)

@router.get("/test", response_model=TestEndpointResponse)
async def test_endpoint(
    date: Optional[datetime.date] = None,
    tags: Optional[List[str]] = Query(default=None, description="Filter by tag names")
):
    return {
        "test": "success",
        "date": str(date) if date else None,
        "tags": tags or []
    }
