import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from database import database
from models import UserProfile, files, search_histories, file_summaries, search_history_summaries, file_upload_webhooks, search_history_upload_webhooks
from dependencies import get_current_user

router = APIRouter(tags=["data"])

# ファイル一覧(日付指定) 例: /files?date=2025-08-18
@router.get("/files")
async def list_files(date: Optional[str] = Query(None, description="YYYY-MM-DD"), current_user: UserProfile = Depends(get_current_user)):
    target_date = datetime.date.today() if date is None else datetime.date.fromisoformat(date)
    start_dt = datetime.datetime.combine(target_date, datetime.time.min)
    end_dt = datetime.datetime.combine(target_date, datetime.time.max)
    q = files.select().with_only_columns(files.c.id).where(
        (files.c.user_id == current_user.id) & (files.c.created_at >= start_dt) & (files.c.created_at <= end_dt)
    )
    rows = await database.fetch_all(q)
    return {"date": target_date.isoformat(), "file_ids": [r[0] for r in rows]}

# ファイル閲覧
@router.get("/file/{file_id}")
async def view_file(file_id: int, current_user: UserProfile = Depends(get_current_user)):
    q = files.select().where((files.c.id == file_id) & (files.c.user_id == current_user.id))
    row = await database.fetch_one(q)
    if not row:
        raise HTTPException(status_code=404, detail="File not found")
    return {"file_id": row["id"], "path": row["file_path"], "created_at": row["created_at"].isoformat() if row["created_at"] else None}

# 閲覧履歴
@router.get("/history/{date}")
async def view_history(date: str, current_user: UserProfile = Depends(get_current_user)):
    target_date = datetime.date.fromisoformat(date)
    start_dt = datetime.datetime.combine(target_date, datetime.time.min)
    end_dt = datetime.datetime.combine(target_date, datetime.time.max)
    q = search_histories.select().where(
        (search_histories.c.user_id == current_user.id) & (search_histories.c.created_at >= start_dt) & (search_histories.c.created_at <= end_dt)
    )
    rows = await database.fetch_all(q)
    return {"date": date, "histories": [{"id": r["id"], "query": r["query"], "created_at": r["created_at"].isoformat() if r["created_at"] else None} for r in rows]}

# 生データログ
@router.get("/log/{date}")
async def get_log(date: str, type: Optional[str] = Query(None, description="file|history"), current_user: UserProfile = Depends(get_current_user)):
    target_date = datetime.date.fromisoformat(date)
    start_dt = datetime.datetime.combine(target_date, datetime.time.min)
    end_dt = datetime.datetime.combine(target_date, datetime.time.max)
    results = {}
    if type in (None, "file"):
        fq = file_upload_webhooks.select().where(
            (file_upload_webhooks.c.created_at >= start_dt) & (file_upload_webhooks.c.created_at <= end_dt)
        )
        frows = await database.fetch_all(fq)
        results["file_webhooks"] = [
            {"id": r["id"], "external_id": r["external_id"], "created_at": r["created_at"].isoformat()} for r in frows
        ]
    if type in (None, "history"):
        hq = search_history_upload_webhooks.select().where(
            (search_history_upload_webhooks.c.created_at >= start_dt) & (search_history_upload_webhooks.c.created_at <= end_dt)
        )
        hrows = await database.fetch_all(hq)
        results["history_webhooks"] = [
            {"id": r["id"], "external_id": r["external_id"], "created_at": r["created_at"].isoformat()} for r in hrows
        ]
    return {"date": date, **results}

# AI処理済みデータアーカイブ
@router.get("/archive/{date}")
async def get_archive(date: str, type: Optional[str] = Query(None, description="file|history"), current_user: UserProfile = Depends(get_current_user)):
    target_date = datetime.date.fromisoformat(date)
    start_dt = datetime.datetime.combine(target_date, datetime.time.min)
    end_dt = datetime.datetime.combine(target_date, datetime.time.max)
    result = {}
    if type in (None, "file"):
        fq = file_summaries.select().where(file_summaries.c.created_at.between(start_dt, end_dt))
        frows = await database.fetch_all(fq)
        result["file_summaries"] = [
            {"id": r["id"], "file_id": r["file_id"], "created_at": r["created_at"].isoformat()} for r in frows
        ]
    if type in (None, "history"):
        hq = search_history_summaries.select().where(search_history_summaries.c.created_at.between(start_dt, end_dt))
        hrows = await database.fetch_all(hq)
        result["history_summaries"] = [
            {"id": r["id"], "search_history_id": r["search_history_id"], "created_at": r["created_at"].isoformat()} for r in hrows
        ]
    return {"date": date, **result}
