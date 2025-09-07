"""ユーザー関連エンドポイント。"""

import datetime
import uuid
from datetime import timedelta

from fastapi import APIRouter, Depends, Query, status
from errors import bad_request, unauthorized

from database import database
from dependencies import get_current_user
from models import (
    CreateAccountResponse,
    RootResponse,
    RefreshRequest,
    RefreshResponse,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserProfile,
    UserProfileActivity,
    file_upload_webhooks,
    files,
    refresh_tokens,
    search_histories,
    users,
)
from security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
    create_access_token,
    generate_refresh_token,
    get_password_hash,
    hash_token,
    verify_password,
)
from utils.datetime_utils import naive_utc_now, month_range

router = APIRouter(tags=["users"])


@router.post("/create", response_model=CreateAccountResponse, operation_id="createAccount")
async def create_account(user: UserCreate) -> CreateAccountResponse:
    query = users.select().where(users.c.email == user.email)
    if await database.fetch_one(query):
        bad_request("Email already registered")
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
    return CreateAccountResponse(
        message="Account created successfully",
        user_id=user_id,
        email=user.email,
        webhooks={"file_external_id": file_external_id},
    )

@router.post("/login", response_model=TokenResponse, operation_id="login")
async def login(user: UserLogin) -> TokenResponse:
    query = users.select().where(users.c.email == user.email)
    db_user = await database.fetch_one(query)

    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        unauthorized("Incorrect email or password", headers={"WWW-Authenticate": "Bearer"})

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

    return TokenResponse(access_token=access_token, token_type="bearer", refresh_token=rt)


@router.post("/refresh", response_model=RefreshResponse, operation_id="refreshToken")
async def refresh_token(payload: RefreshRequest) -> RefreshResponse:
    token = payload.refresh_token
    token_h = hash_token(token)
    row = await database.fetch_one(
        refresh_tokens.select().where(refresh_tokens.c.token_hash == token_h)
    )
    if not row or row["revoked"]:
        unauthorized("Invalid refresh token")
    now = naive_utc_now()
    if row["expires_at"] < now:
        unauthorized("Refresh token expired")

    # Fetch user
    user_row = await database.fetch_one(users.select().where(users.c.id == row["user_id"]))
    if not user_row:
        unauthorized("User not found")

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
    return RefreshResponse(access_token=access_token, refresh_token=new_rt)

@router.post("/logout", response_model=RootResponse, operation_id="logout")
async def logout(payload: RefreshRequest) -> RootResponse:
    token_h = hash_token(payload.refresh_token)
    res = await database.execute(
        refresh_tokens.update()
        .where(refresh_tokens.c.token_hash == token_h)
        .values(revoked=True)
    )
    # No need to expose details; return ok regardless
    return RootResponse(message="ok")

@router.get("/profile", response_model=UserProfileActivity, operation_id="getProfile")
async def get_profile(
    month: str | None = Query(None, description="YYYY-MM"),
    current_user: UserProfile = Depends(get_current_user),
) -> UserProfileActivity:
    now = naive_utc_now()
    if month:
        try:
            target_year, target_month = map(int, month.split("-"))
        except ValueError:
            bad_request("Invalid month format. Use YYYY-MM")
    else:
        target_year, target_month = now.year, now.month
    # 対象月の範囲
    month_start, next_month_start = month_range(target_year, target_month)

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
