"""テスト用の軽量エンドポイント。"""

import datetime
from typing import List, Optional

from fastapi import APIRouter, Query

from models import TestEndpointResponse

router = APIRouter(tags=["test"])


@router.get("/test", response_model=TestEndpointResponse, operation_id="testEndpoint")
async def test_endpoint(
    date: Optional[datetime.date] = None,
    tags: Optional[List[str]] = Query(default=None, description="タグ名でフィルタ")
) -> TestEndpointResponse:
    return TestEndpointResponse(test="success", date=str(date) if date else None, tags=tags or [])
