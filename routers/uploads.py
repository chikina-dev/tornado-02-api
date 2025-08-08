from fastapi import APIRouter, Depends, UploadFile, File
from models import UserProfile
from dependencies import get_current_user

router = APIRouter()

# ファイルアップロード (認証不要)(もうちょっと変えるかも)
@router.post("/upload/file")
async def upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename, "content_type": file.content_type}


# 検索履歴アップロード (認証必須)
@router.post("/upload/history")
async def upload_history(history_data: dict, current_user: UserProfile = Depends(get_current_user)):
    return {"message": "History uploaded successfully", "user": current_user.email, "data_received": history_data}
