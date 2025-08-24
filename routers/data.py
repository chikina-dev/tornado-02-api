import datetime
import os
import base64
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from database import database
from models import (
    UserProfile,
    FilesListResponse,
    HistoryListResponse,
    LogResponse,
    ArchiveResponse,
    FileBinaryResponse,
    files,
    search_histories,
    file_summaries,
    file_upload_webhooks,
)
from utils.datetime_utils import naive_utc_now, parse_ymd_date
from dependencies import get_current_user

router = APIRouter(tags=["data"])

# ファイル一覧(日付指定) 例: /files?date=2025-08-18
@router.get("/files", response_model=FilesListResponse)
async def list_files(date: Optional[str] = Query(None, description="YYYY-MM-DD"), current_user: UserProfile = Depends(get_current_user)):
    if date is None:
        local_now = naive_utc_now()
        target_date = local_now.date()
    else:
        try:
            target_date = parse_ymd_date(date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    start_dt = datetime.datetime.combine(target_date, datetime.time.min)
    end_dt = datetime.datetime.combine(target_date, datetime.time.max)
    q = files.select().with_only_columns(files.c.id).where(
        (files.c.user_id == current_user.id) & (files.c.created_at >= start_dt) & (files.c.created_at <= end_dt)
    )
    rows = await database.fetch_all(q)
    return {"date": target_date.isoformat(), "file_ids": [r[0] for r in rows]}

# ファイル閲覧: ファイルパスではなく実ファイルを返す
@router.get("/file/{file_id}", response_model=FileBinaryResponse)
async def view_file(file_id: int, current_user: UserProfile = Depends(get_current_user)):
    q = files.select().where((files.c.id == file_id) & (files.c.user_id == current_user.id))
    row = await database.fetch_one(q)
    if not row:
        raise HTTPException(status_code=404, detail="File not found")
    file_path = row["file_path"]
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not available")
    # Read file and return base64 content in JSON
    base = os.path.basename(file_path)
    filename = base.split("_", 1)[1] if "_" in base else base
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # naive content type guess
    content_type = None
    if "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext in {"txt", "log", "md"}:
            content_type = "text/plain"
        elif ext in {"jpg", "jpeg"}:
            content_type = "image/jpeg"
        elif ext == "png":
            content_type = "image/png"
        elif ext == "pdf":
            content_type = "application/pdf"
    return {
        "file_id": row["id"],
        "filename": filename,
        "content_base64": b64,
        "content_type": content_type,
        "created_at": row["created_at"],
    }

# 閲覧履歴
@router.get("/history/{date}", response_model=HistoryListResponse)
async def view_history(date: str, current_user: UserProfile = Depends(get_current_user)):
    try:
        target_date = parse_ymd_date(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    start_dt = datetime.datetime.combine(target_date, datetime.time.min)
    end_dt = datetime.datetime.combine(target_date, datetime.time.max)
    q = search_histories.select().where(
        (search_histories.c.user_id == current_user.id) & (search_histories.c.created_at >= start_dt) & (search_histories.c.created_at <= end_dt)
    )
    rows = await database.fetch_all(q)
    return {
        "date": date,
        "histories": [
            {
                "id": r["id"],
                "url": r["url"],
                "title": r["title"],
                "description": r["description"],
                "created_at": r["created_at"],
            }
            for r in rows
        ],
    }

# 生データログ
@router.get("/log/{date}", response_model=LogResponse)
async def get_log(date: str, type: Optional[str] = Query(None, description="file|history"), current_user: UserProfile = Depends(get_current_user)):
    try:
        target_date = parse_ymd_date(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    start_dt = datetime.datetime.combine(target_date, datetime.time.min)
    end_dt = datetime.datetime.combine(target_date, datetime.time.max)
    results = {}
    if type in (None, "file"):
        fq = file_upload_webhooks.select().where(
            (file_upload_webhooks.c.created_at >= start_dt) & (file_upload_webhooks.c.created_at <= end_dt)
        )
        frows = await database.fetch_all(fq)
        results["file_webhooks"] = [
            {"id": r["id"], "external_id": r["external_id"], "created_at": r["created_at"]} for r in frows
        ]
    # history webhooks are removed
    return {"date": date, **results}

# AI処理済みデータアーカイブ
@router.get("/archive/{date}", response_model=ArchiveResponse)
async def get_archive(date: str, type: Optional[str] = Query(None, description="file|history"), current_user: UserProfile = Depends(get_current_user)):
    try:
        target_date = parse_ymd_date(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    start_dt = datetime.datetime.combine(target_date, datetime.time.min)
    end_dt = datetime.datetime.combine(target_date, datetime.time.max)
    result = {}
    if type in (None, "file"):
        fq = file_summaries.select().where(file_summaries.c.created_at.between(start_dt, end_dt))
        frows = await database.fetch_all(fq)
        result["file_summaries"] = [
            {"id": r["id"], "file_id": r["file_id"], "created_at": r["created_at"]} for r in frows
        ]
    # history summaries are consolidated into daily_summaries; omit here
    return {"date": date, **result}
