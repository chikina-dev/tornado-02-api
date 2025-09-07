"""データ取得系エンドポイント。"""

import datetime
import os
from typing import Optional

from fastapi import APIRouter, Depends, Query
from errors import bad_request, not_found

from database import database
from dependencies import get_current_user
from models import (
    ArchiveResponse,
    FileBinaryResponse,
    FilesListResponse,
    HistoryListResponse,
    LogResponse,
    UserProfile,
    file_summaries,
    file_upload_webhooks,
    files,
    search_histories,
)
from utils.datetime_utils import day_range, naive_utc_now, parse_ymd_date
from utils.file_utils import display_filename, guess_content_type, read_file_as_base64

router = APIRouter(tags=["data"])

# ファイル一覧（日付指定）例: /files?date=2025-08-18
@router.get("/files", response_model=FilesListResponse, operation_id="listFilesByDate")
async def list_files(
    date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    current_user: UserProfile = Depends(get_current_user),
) -> FilesListResponse:
    if date is None:
        local_now = naive_utc_now()
        target_date = local_now.date()
    else:
        try:
            target_date = parse_ymd_date(date)
        except ValueError:
            bad_request("Invalid date format. Use YYYY-MM-DD")
    start_dt, end_dt = day_range(target_date)
    q = files.select().with_only_columns(files.c.id).where(
        (files.c.user_id == current_user.id) & (files.c.created_at >= start_dt) & (files.c.created_at <= end_dt)
    )
    rows = await database.fetch_all(q)
    return FilesListResponse(date=target_date.isoformat(), file_ids=[r[0] for r in rows])

# ファイル閲覧（実体をBase64で返す）
@router.get("/file/{file_id}", response_model=FileBinaryResponse, operation_id="getFileById")
async def view_file(file_id: int, current_user: UserProfile = Depends(get_current_user)) -> FileBinaryResponse:
    q = files.select().where((files.c.id == file_id) & (files.c.user_id == current_user.id))
    row = await database.fetch_one(q)
    if not row:
        not_found("File not found")
    file_path = row["file_path"]
    if not file_path or not os.path.exists(file_path):
        not_found("File not available")
    filename = display_filename(file_path)
    b64 = read_file_as_base64(file_path)
    content_type = guess_content_type(filename)
    return FileBinaryResponse(
        file_id=row["id"],
        filename=filename,
        content_base64=b64,
        content_type=content_type,
        created_at=row["created_at"],
    )

# 閲覧履歴一覧
@router.get("/history/{date}", response_model=HistoryListResponse, operation_id="listHistoryByDate")
async def view_history(date: str, current_user: UserProfile = Depends(get_current_user)) -> HistoryListResponse:
    try:
        target_date = parse_ymd_date(date)
    except ValueError:
        bad_request("Invalid date format. Use YYYY-MM-DD")
    start_dt, end_dt = day_range(target_date)
    q = search_histories.select().where(
        (search_histories.c.user_id == current_user.id) & (search_histories.c.created_at >= start_dt) & (search_histories.c.created_at <= end_dt)
    )
    from models import HistoryItem
    rows = await database.fetch_all(q)
    return HistoryListResponse(
        date=date,
        histories=[
            HistoryItem(
                id=r["id"], url=r["url"], title=r["title"], description=r["description"], created_at=r["created_at"]
            )
            for r in rows
        ],
    )

# 生データログ
@router.get("/log/{date}", response_model=LogResponse, operation_id="getLogByDate")
async def get_log(
    date: str,
    kind: Optional[str] = Query(None, alias="type", description="file|history"),
    current_user: UserProfile = Depends(get_current_user),
) -> LogResponse:
    try:
        target_date = parse_ymd_date(date)
    except ValueError:
        bad_request("Invalid date format. Use YYYY-MM-DD")
    start_dt, end_dt = day_range(target_date)
    results = {}
    if kind in (None, "file"):
        fq = file_upload_webhooks.select().where(
            (file_upload_webhooks.c.created_at >= start_dt) & (file_upload_webhooks.c.created_at <= end_dt)
        )
        frows = await database.fetch_all(fq)
        from models import FileWebhookLogItem
        results["file_webhooks"] = [
            FileWebhookLogItem(id=r["id"], external_id=r["external_id"], created_at=r["created_at"]) for r in frows
        ]
    # history webhook は撤廃済み
    return LogResponse(date=date, **results)

# AI処理済みデータのアーカイブ
@router.get("/archive/{date}", response_model=ArchiveResponse, operation_id="getArchiveByDate")
async def get_archive(
    date: str,
    type: Optional[str] = Query(None, description="file|history"),
    current_user: UserProfile = Depends(get_current_user),
) -> ArchiveResponse:
    try:
        target_date = parse_ymd_date(date)
    except ValueError:
        bad_request("Invalid date format. Use YYYY-MM-DD")
    start_dt = datetime.datetime.combine(target_date, datetime.time.min)
    end_dt = datetime.datetime.combine(target_date, datetime.time.max)
    result = {}
    if type in (None, "file"):
        fq = file_summaries.select().where(file_summaries.c.created_at.between(start_dt, end_dt))
        frows = await database.fetch_all(fq)
        from models import ArchiveFileSummaryItem
        result["file_summaries"] = [
            ArchiveFileSummaryItem(id=r["id"], file_id=r["file_id"], created_at=r["created_at"]) for r in frows
        ]
    # history要約はdaily_summariesに集約済みのため省略
    return ArchiveResponse(date=date, **result)
