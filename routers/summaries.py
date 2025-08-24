from fastapi import APIRouter, Depends, HTTPException
import datetime

from database import database
from models import UserProfile, daily_summaries, tags, daily_summary_tags
from dependencies import get_current_user

router = APIRouter(prefix="/summaries", tags=["summaries"])


@router.get("/{year:int}/{month:int}/{day:int}")
async def get_daily_summary_by_md(
    year: int,
    month: int,
    day: int,
    current_user: UserProfile = Depends(get_current_user),
):
    try:
        target_date = datetime.date(year, month, day)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid month/day")

    row = await database.fetch_one(
        daily_summaries.select().where(
            (daily_summaries.c.user_id == current_user.id) & (daily_summaries.c.date == target_date)
        )
    )
    if not row:
        raise HTTPException(status_code=404, detail="Summary not found")
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
    return {
        "date": target_date.isoformat(),
        "markdown": row["markdown"],
        "created_at": row["created_at"],
        "tags": [r["name"] for r in trows],
    }
