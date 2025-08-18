from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from database import database
from models import UserProfile, Tag, tags, search_history_tags, search_histories
from dependencies import get_current_user

router = APIRouter(prefix="/tags", tags=["tags"])

# タグ一覧取得
@router.get("/", response_model=List[Tag])
async def list_tags(current_user: UserProfile = Depends(get_current_user)):
    query = tags.select().where(tags.c.user_id == current_user.id).order_by(tags.c.name.asc())
    rows = await database.fetch_all(query)
    return [Tag(id=r["id"], name=r["name"]) for r in rows]

# タグを元にした検索
@router.get("/search_history/{search_history_id}", response_model=List[Tag])
async def list_tags_for_search_history(search_history_id: int, current_user: UserProfile = Depends(get_current_user)):
    sh_query = search_histories.select().where(search_histories.c.id == search_history_id, search_histories.c.user_id == current_user.id)
    sh = await database.fetch_one(sh_query)
    if not sh:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Search history not found")
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
