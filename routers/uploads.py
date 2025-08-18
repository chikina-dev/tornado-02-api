import os
import shutil
import datetime
from datetime import timezone
from fastapi import APIRouter, Depends, UploadFile, File, Path
from models import UserProfile, files, search_histories, file_upload_webhooks, search_history_upload_webhooks
from dependencies import get_current_user
from database import database

UPLOAD_DIR = os.path.join(os.getcwd(), "uploaded_files")
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter(tags=["upload"])

# ファイルアップロード 
@router.post("/upload/file")
async def upload_file(file: UploadFile = File(...), current_user: UserProfile = Depends(get_current_user)):
    """ユーザー自身によるファイルアップロード"""
    now = datetime.datetime.now(timezone.utc)
    saved_path = os.path.join(UPLOAD_DIR, f"{now.timestamp()}_{file.filename}")
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    insert_q = files.insert().values(user_id=current_user.id, file_path=saved_path, created_at=now)
    file_id = await database.execute(insert_q)
    return {"file_id": file_id, "filename": file.filename}


# 検索履歴アップロード (認証必須)
@router.post("/upload/history")
async def upload_history(history_data: dict, current_user: UserProfile = Depends(get_current_user)):
    """ユーザー自身による検索履歴アップロード"""
    now = datetime.datetime.now(timezone.utc)
    insert_q = search_histories.insert().values(user_id=current_user.id, query=history_data.get("query", ""), created_at=now)
    history_id = await database.execute(insert_q)
    return {"history_id": history_id}

# webhook用のエンドポイント (認証なし、パスパラメータでユーザーID指定)
@router.post("/upload/file/{user_id}")
async def webhook_file_upload(user_id: int = Path(..., description="ユーザーID"), file: UploadFile = File(...)):
    now = datetime.datetime.now(timezone.utc)
    saved_path = os.path.join(UPLOAD_DIR, f"{now.timestamp()}_{file.filename}")
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    file_id = await database.execute(files.insert().values(user_id=user_id, file_path=saved_path, created_at=now))
    await database.execute(
        file_upload_webhooks.insert().values(user_id=user_id, external_id=file.filename, created_at=now)
    )
    return {"file_id": file_id, "saved": True}

@router.post("/upload/history/{user_id}")
async def webhook_history_upload(history_data: dict, user_id: int = Path(..., description="ユーザーID")):
    now = datetime.datetime.now(timezone.utc)
    history_id = await database.execute(
        search_histories.insert().values(user_id=user_id, query=history_data.get("query", ""), created_at=now)
    )
    await database.execute(
        search_history_upload_webhooks.insert().values(
            user_id=user_id,
            external_id=history_data.get("external_id", str(history_id)),
            created_at=now,
        )
    )
    return {"history_id": history_id, "saved": True}
