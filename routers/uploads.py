import os
import shutil
from fastapi import APIRouter, Depends, UploadFile, File, Path
from models import (
    HistoryPayload,
    UserProfile,
    UploadFileResponse,
    UploadHistoryResponse,
    WebhookUploadFileResponse,
    files,
    search_histories,
    file_upload_webhooks,
)
from dependencies import get_current_user
from database import database
from utils.datetime_utils import naive_utc_now, as_naive_utc

UPLOAD_DIR = os.path.join(os.getcwd(), "uploaded_files")
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter(tags=["upload"])

# ファイルアップロード 
@router.post("/upload/file", response_model=UploadFileResponse)
async def upload_file(file: UploadFile = File(...), current_user: UserProfile = Depends(get_current_user)):
    now = naive_utc_now()
    saved_path = os.path.join(UPLOAD_DIR, f"{now.timestamp()}_{file.filename}")
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    insert_q = files.insert().values(user_id=current_user.id, file_path=saved_path, created_at=now)
    file_id = await database.execute(insert_q)
    return {"file_id": file_id, "filename": file.filename}


# 検索履歴アップロード (認証必須)
@router.post("/upload/history", response_model=UploadHistoryResponse)
async def upload_history(history: HistoryPayload, current_user: UserProfile = Depends(get_current_user)):
    created_at = as_naive_utc(history.timestamp)
    values = {
        "user_id": current_user.id,
        "url": history.url,
        "title": history.title,
        "description": history.description,
        "created_at": created_at,
    }
    history_id = await database.execute(search_histories.insert().values(**values))
    return {"history_id": history_id}

# webhook用のエンドポイント (認証なし、パスパラメータでユーザーID指定)
@router.post("/upload/file/{user_id}", response_model=WebhookUploadFileResponse)
async def webhook_file_upload(user_id: int = Path(..., description="ユーザーID"), file: UploadFile = File(...)):
    now = naive_utc_now()
    saved_path = os.path.join(UPLOAD_DIR, f"{now.timestamp()}_{file.filename}")
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    file_id = await database.execute(files.insert().values(user_id=user_id, file_path=saved_path, created_at=now))
    await database.execute(
        file_upload_webhooks.insert().values(user_id=user_id, external_id=file.filename, created_at=now)
    )
    return {"file_id": file_id, "saved": True}
