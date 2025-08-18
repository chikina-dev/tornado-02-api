from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from datetime import timedelta
import datetime
from datetime import timezone

from database import database
from models import (
    UserCreate,
    UserLogin,
    UserProfile,
    UserProfileActivity,
    users,
    files,
    search_histories,
    file_upload_webhooks,
    search_history_upload_webhooks,
)
import uuid
from dependencies import get_current_user
from security import create_access_token, verify_password, get_password_hash, ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter(tags=["users"])

# アカウント作成
@router.post("/create")
async def create_account(user: UserCreate):
    query = users.select().where(users.c.email == user.email)
    if await database.fetch_one(query):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    hashed_password = get_password_hash(user.password)
    query = users.insert().values(email=user.email, hashed_password=hashed_password)
    user_id = await database.execute(query)
    file_external_id = f"file-{uuid.uuid4()}"
    history_external_id = f"history-{uuid.uuid4()}"
    now = datetime.datetime.now(timezone.utc)
    await database.execute(
        file_upload_webhooks.insert().values(
            user_id=user_id,
            external_id=file_external_id,
            created_at=now,
        )
    )
    await database.execute(
        search_history_upload_webhooks.insert().values(
            user_id=user_id,
            external_id=history_external_id,
            created_at=now,
        )
    )
    return {"message": "Account created successfully", "user_id": user_id, "email": user.email, "webhooks": {"file_external_id": file_external_id, "history_external_id": history_external_id}}

# ログイン
@router.post("/login")
async def login(user: UserLogin):
    query = users.select().where(users.c.email == user.email)
    db_user = await database.fetch_one(query)

    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user["email"], "id": db_user["id"]}, expires_delta=access_token_expires
    )

    response = JSONResponse(content={"message": "Login successful"})
    response.set_cookie(
        key="token",
        value=access_token,
        httponly=True,
        max_age=int(access_token_expires.total_seconds()),
        samesite="lax",
        path="/",
    )
    return response

@router.get("/profile", response_model=UserProfileActivity)
async def get_profile(
    month: str | None = Query(None, description="YYYY-MM"),
    current_user: UserProfile = Depends(get_current_user),
):
    now = datetime.datetime.now(timezone.utc)
    if month:
        try:
            target_year, target_month = map(int, month.split("-"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid month format. Use YYYY-MM")
    else:
        target_year, target_month = now.year, now.month
    # 月初・翌月初
    month_start = datetime.datetime(target_year, target_month, 1, tzinfo=timezone.utc)
    if target_month == 12:
        next_month_start = datetime.datetime(target_year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        next_month_start = datetime.datetime(target_year, target_month + 1, 1, tzinfo=timezone.utc)

    # 集計クエリ (日単位)
    file_q = (
        files.select()
        .with_only_columns(files.c.created_at)
        .where(
            (files.c.user_id == current_user.id)
            & (files.c.created_at >= month_start)
            & (files.c.created_at < next_month_start)
        )
    )
    history_q = (
        search_histories.select()
        .with_only_columns(search_histories.c.created_at)
        .where(
            (search_histories.c.user_id == current_user.id)
            & (search_histories.c.created_at >= month_start)
            & (search_histories.c.created_at < next_month_start)
        )
    )
    file_rows = await database.fetch_all(file_q)
    history_rows = await database.fetch_all(history_q)

    file_days = {r[0].day for r in file_rows if r[0]}
    history_days = {r[0].day for r in history_rows if r[0]}
    active_days = sorted(file_days | history_days)

    month_repr = f"{target_year:04d}-{target_month:02d}"
    return UserProfileActivity(
        id=current_user.id,
        email=current_user.email,
        month=month_repr,
        active_dates=active_days,
        file_active_dates=sorted(file_days) if file_days else [],
        history_active_dates=sorted(history_days) if history_days else [],
    )
