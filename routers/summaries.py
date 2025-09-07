"""サマリー取得エンドポイント。"""

import datetime

from fastapi import APIRouter, Depends
from errors import bad_request, not_found

from database import database
from dependencies import get_current_user
from models import DailySummaryResponse, AnalysisResponse, UserProfile, daily_summaries, daily_summary_tags, tags
from services.daily_summary import analyze_day as analyze_service

router = APIRouter(tags=["summaries"])  # No prefix; absolute paths


@router.get("/summaries/{year:int}/{month:int}/{day:int}", response_model=DailySummaryResponse, operation_id="getDailySummaryByDate")
async def get_daily_summary_by_md(
    year: int,
    month: int,
    day: int,
    current_user: UserProfile = Depends(get_current_user),
) -> DailySummaryResponse:
    try:
        target_date = datetime.date(year, month, day)
    except Exception:
        bad_request("Invalid month/day")

    row = await database.fetch_one(
        daily_summaries.select().where(
            (daily_summaries.c.user_id == current_user.id) & (daily_summaries.c.date == target_date)
        )
    )
    if not row:
        not_found("Summary not found")
    # fetch tags for this daily summary
    join_q = tags.join(daily_summary_tags, tags.c.id == daily_summary_tags.c.tag_id)
    trows = await database.fetch_all(
        tags.select()
        .select_from(join_q)
        .where(
            (daily_summary_tags.c.daily_summary_id == row["id"]) & (tags.c.user_id == current_user.id)
        )
        .order_by(tags.c.name.asc())
    )
    return DailySummaryResponse(
        date=target_date.isoformat(),
        markdown=row["markdown"],
        created_at=row["created_at"],
        tags=[r["name"] for r in trows],
    )


@router.get("/analysis/{year:int}/{month:int}/{day:int}", response_model=AnalysisResponse, operation_id="getDailyAnalysisByDate")
async def get_daily_analysis_by_date(
    year: int,
    month: int,
    day: int,
    current_user: UserProfile = Depends(get_current_user),
) -> AnalysisResponse:
    try:
        target_date = datetime.date(year, month, day)
    except Exception:
        bad_request("Invalid month/day")

    result = await analyze_service(current_user.id, target_date)
    return AnalysisResponse(user_id=current_user.id, date=target_date.isoformat(), **result)
