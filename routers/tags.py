"""タグ関連エンドポイント。"""

from typing import List

from fastapi import APIRouter, Depends, status
from errors import not_found

from database import database
from dependencies import get_current_user
from models import (
    Tag,
    UserProfile,
    daily_summaries,
    daily_summary_tags,
    search_histories,
    search_history_tags,
    tags,
)

router = APIRouter(prefix="/tags", tags=["tags"])

@router.get("/", response_model=List[Tag], operation_id="listTags")
async def list_tags(current_user: UserProfile = Depends(get_current_user)) -> List[Tag]:
    query = tags.select().where(tags.c.user_id == current_user.id).order_by(tags.c.name.asc())
    rows = await database.fetch_all(query)
    return [Tag(id=r["id"], name=r["name"]) for r in rows]

@router.get("/search_history/{search_history_id}", response_model=List[Tag], operation_id="listTagsForSearchHistory")
async def list_tags_for_search_history(
    search_history_id: int, current_user: UserProfile = Depends(get_current_user)
) -> List[Tag]:
    sh_query = search_histories.select().where(search_histories.c.id == search_history_id, search_histories.c.user_id == current_user.id)
    sh = await database.fetch_one(sh_query)
    if not sh:
        not_found("Search history not found")
    join_q = (
        tags.join(search_history_tags, tags.c.id == search_history_tags.c.tag_id)
    )
    query = (
        tags.select()
        .select_from(join_q)
        .where(search_history_tags.c.search_history_id == search_history_id, tags.c.user_id == current_user.id)
    )
    rows = await database.fetch_all(query)
    return [Tag(id=r["id"], name=r["name"]) for r in rows]

@router.get("/daily/{date}", response_model=List[Tag], operation_id="listTagsForDaily")
async def list_tags_for_daily(date: str, current_user: UserProfile = Depends(get_current_user)) -> List[Tag]:
    ds = await database.fetch_one(
        daily_summaries.select().where(
            (daily_summaries.c.user_id == current_user.id) & (daily_summaries.c.date == date)
        )
    )
    if not ds:
        return []
    join_q = tags.join(daily_summary_tags, tags.c.id == daily_summary_tags.c.tag_id)
    q = (
        tags.select()
        .select_from(join_q)
        .where(daily_summary_tags.c.daily_summary_id == ds["id"], tags.c.user_id == current_user.id)
        .order_by(tags.c.name.asc())
    )
    rows = await database.fetch_all(q)
    return [Tag(id=r["id"], name=r["name"]) for r in rows]
