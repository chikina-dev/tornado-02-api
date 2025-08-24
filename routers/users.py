from fastapi import APIRouter, Depends, HTTPException, status, Query
from datetime import timedelta
import datetime

from database import database
from models import (
    UserCreate,
    UserLogin,
    UserProfile,
    UserProfileActivity,
    CreateAccountResponse,
    TokenResponse,
    RefreshRequest,
    RefreshResponse,
    users,
    files,
    search_histories,
    file_upload_webhooks,
    refresh_tokens,
)
import uuid
from dependencies import get_current_user
from security import (
    create_access_token,
    verify_password,
    get_password_hash,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
    generate_refresh_token,
    hash_token,
)
from utils.datetime_utils import naive_utc_now

router = APIRouter(tags=["users"])

# アカウント作成
@router.post("/create", response_model=CreateAccountResponse)
async def create_account(user: UserCreate):
    query = users.select().where(users.c.email == user.email)
    if await database.fetch_one(query):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    now = naive_utc_now()
    hashed_password = get_password_hash(user.password)
    query = users.insert().values(email=user.email, hashed_password=hashed_password, created_at=now)
    user_id = await database.execute(query)
    file_external_id = f"file-{uuid.uuid4()}"
    await database.execute(
        file_upload_webhooks.insert().values(
            user_id=user_id,
            external_id=file_external_id,
            created_at=now,
        )
    )
    return {
        "message": "Account created successfully",
        "user_id": user_id,
        "email": user.email,
        "webhooks": {"file_external_id": file_external_id},
    }

# ログイン
@router.post("/login", response_model=TokenResponse)
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
    # Issue refresh token and store hashed
    rt = generate_refresh_token()
    now = naive_utc_now()
    expires_at = now + datetime.timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    await database.execute(
        refresh_tokens.insert().values(
            user_id=db_user["id"],
            token_hash=hash_token(rt),
            created_at=now,
            expires_at=expires_at,
            revoked=False,
        )
    )

    return {"access_token": access_token, "token_type": "bearer", "refresh_token": rt}


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_token(payload: RefreshRequest):
    token = payload.refresh_token
    token_h = hash_token(token)
    row = await database.fetch_one(
        refresh_tokens.select().where(refresh_tokens.c.token_hash == token_h)
    )
    if not row or row["revoked"]:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    now = naive_utc_now()
    if row["expires_at"] < now:
        raise HTTPException(status_code=401, detail="Refresh token expired")

    # Fetch user
    user_row = await database.fetch_one(users.select().where(users.c.id == row["user_id"]))
    if not user_row:
        raise HTTPException(status_code=401, detail="User not found")

    access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_row["email"], "id": user_row["id"]},
        expires_delta=access_token_expires,
    )
    # Rotate refresh token: revoke the old one and issue a new one
    await database.execute(
        refresh_tokens.update()
        .where(refresh_tokens.c.token_hash == token_h)
        .values(revoked=True)
    )
    new_rt = generate_refresh_token()
    now2 = naive_utc_now()
    expires_at2 = now2 + datetime.timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    await database.execute(
        refresh_tokens.insert().values(
            user_id=user_row["id"],
            token_hash=hash_token(new_rt),
            created_at=now2,
            expires_at=expires_at2,
            revoked=False,
        )
    )
    return {"access_token": access_token, "refresh_token": new_rt}


@router.post("/logout")
async def logout(payload: RefreshRequest):
    token_h = hash_token(payload.refresh_token)
    res = await database.execute(
        refresh_tokens.update()
        .where(refresh_tokens.c.token_hash == token_h)
        .values(revoked=True)
    )
    # No need to expose details; return ok regardless
    return {"message": "ok"}

@router.get("/profile", response_model=UserProfileActivity)
async def get_profile(
    month: str | None = Query(None, description="YYYY-MM"),
    current_user: UserProfile = Depends(get_current_user),
):
    now = naive_utc_now()
    if month:
        try:
            target_year, target_month = map(int, month.split("-"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid month format. Use YYYY-MM")
    else:
        target_year, target_month = now.year, now.month
    # 月初・翌月初
    month_start = datetime.datetime(target_year, target_month, 1)
    if target_month == 12:
        next_month_start = datetime.datetime(target_year + 1, 1, 1)
    else:
        next_month_start = datetime.datetime(target_year, target_month + 1, 1)

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
