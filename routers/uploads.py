"""アップロード関連のエンドポイント。"""

import os
import shutil

from fastapi import APIRouter, Depends, File, HTTPException, Path, UploadFile, Request

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

UPLOAD_DIR = os.path.join(os.getcwd(), "uploaded_files")
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

@router.post(
    "/upload/file/{user_id}",
    response_model=WebhookUploadFileResponse,
    operation_id="uploadFileByUserId",
)
async def webhook_file_upload(request: Request, user_id: int = Path(...)):
    ct = request.headers.get("content-type", "").lower()
    upload_obj: UploadFile | None = None
    raw: bytes | None = None
    now = naive_utc_now()
    print(raw)

    # 保存用のパス
    def build_saved_path(filename: str) -> str:
        safe_name = filename or "upload.bin"
        return os.path.join(UPLOAD_DIR, f"{now.timestamp()}_{safe_name}")

    if ct.startswith("multipart/form-data"):
        form = await request.form()
        for key, val in form.items():
            if isinstance(val, UploadFile):
                upload_obj = val
                break
        if upload_obj is None:
            raise HTTPException(status_code=422, detail="No file in multipart form data")

        saved_path = build_saved_path(upload_obj.filename)
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(upload_obj.file, f)
        external_id = upload_obj.filename

    else:
        raw = await request.body()
        if not raw:
            raise HTTPException(status_code=422, detail="Empty request body")
        filename = request.headers.get("x-filename") or request.headers.get("x-file-name") or "upload.bin"
        saved_path = build_saved_path(filename)
        with open(saved_path, "wb") as f:
            f.write(raw)
        external_id = filename

    # DB 反映はそのまま
    file_id = await database.execute(
        files.insert().values(user_id=user_id, file_path=saved_path, created_at=now)
    )
    await database.execute(
        file_upload_webhooks.insert().values(user_id=user_id, external_id=external_id, created_at=now)
    )
    return WebhookUploadFileResponse(file_id=file_id, saved=True)
