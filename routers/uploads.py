"""アップロード関連のエンドポイント。"""

import os
import shutil
from pathlib import Path as FilePath
from typing import List

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, Path, UploadFile, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from database import database
from dependencies import get_current_user
from models import (
    HistoryPayload,
    UploadFileResponse,
    UploadHistoryResponse,
    UserProfile,
    WebhookUploadFileResponse,
    file_upload_webhooks,
    files,
    search_histories,
)
from utils.datetime_utils import as_naive_utc, naive_utc_now

UPLOAD_DIR = FilePath("uploaded_files")
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter(tags=["upload"])


@router.post("/upload/file", response_model=UploadFileResponse, operation_id="uploadFile")
async def upload_file(
    file: UploadFile = File(...),
    current_user: UserProfile = Depends(get_current_user),
):
    now = naive_utc_now()
    saved_path = os.path.join(UPLOAD_DIR, f"{now.timestamp()}_{file.filename}")
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    insert_q = files.insert().values(user_id=current_user.id, file_path=saved_path, created_at=now)
    file_id = await database.execute(insert_q)
    return UploadFileResponse(file_id=file_id, filename=file.filename)


@router.post("/upload/history", response_model=UploadHistoryResponse, operation_id="uploadHistory")
async def upload_history(
    history: HistoryPayload, current_user: UserProfile = Depends(get_current_user)
):
    created_at = as_naive_utc(history.timestamp)
    values = {
        "user_id": current_user.id,
        "url": history.url,
        "title": history.title,
        "description": history.description,
        "created_at": created_at,
    }
    history_id = await database.execute(search_histories.insert().values(**values))
    return UploadHistoryResponse(history_id=history_id)

class WebhookUploadFileResponseItem(BaseModel):
    filename: str
    size: int
    content_preview: str

@router.post("/upload/file/{user_id}")
async def upload_files_inspect(request: Request, user_id: int = Path(...)):
    form = await request.form()  # multipart/form-dataを取得
    keys = list(form.keys())     # 送信されたすべてのフィールド名
    files_info = []

    for key, value in form.items():
        if hasattr(value, "filename"):  # UploadFileかどうかチェック
            file_content = await value.read()
            
            # 保存するパスを生成
            save_path = UPLOAD_DIR / f"{user_id}_{value.filename}"
            
            # 非同期でファイルを書き込み
        async with aiofiles.open(save_path, "wb") as f:
            await f.write(file_content)
            
        files_info.append({
            "field_name": key,
            "filename": value.filename,
            "size": len(file_content),
            "saved_path": str(save_path)
        })
    
    print(f"Received files for user_id={user_id}: {files_info}")

    return JSONResponse({
        "user_id": user_id,
        "keys_received": keys,
        "files_info": files_info,
        "message": f"{len(files_info)} file(s) uploaded successfully"
    })
