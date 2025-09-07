"""FastAPI アプリのエントリポイント。

可読性重視で最小限の設定のみ記述する。
"""

import datetime
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from database import database
from models import RootResponse
from errors import AppError
from routers import data, test, uploads, users
from routers import admin as admin_router
from routers import summaries as summaries_router
from routers import tags as tags_router
from scheduler.summaries import (
    regen_all,
    analyze_all,
    feedback_all,
)
from scheduler.tokens import prune_expired_refresh_tokens

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリ起動/終了時の処理。DB接続とスケジューラのみ。

    - 0:00 前日の要約再生成
    - 1:00 リフレッシュトークンの掃除
    """
    await database.connect()
    scheduler = AsyncIOScheduler(timezone=ZoneInfo("Asia/Tokyo"))

    async def daily_summary_job():
        today = datetime.date.today()
        target_date = today - datetime.timedelta(days=1)
        await regen_all(target_date)

    async def daily_analysis_job():
        today = datetime.date.today()
        target_date = today - datetime.timedelta(days=1)
        await analyze_all(target_date)

    async def daily_feedback_job():
        today = datetime.date.today()
        target_date = today - datetime.timedelta(days=1)
        await feedback_all(target_date)

    # ジョブの衝突を避けるために時刻をずらす
    scheduler.add_job(daily_summary_job, CronTrigger(hour=0, minute=0))
    scheduler.add_job(daily_analysis_job, CronTrigger(hour=0, minute=10))
    scheduler.add_job(daily_feedback_job, CronTrigger(hour=0, minute=20))
    scheduler.add_job(prune_expired_refresh_tokens, CronTrigger(hour=1, minute=0))
    scheduler.start()
    try:
        yield
    finally:
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            pass
        await database.disconnect()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "https://chikina-dev.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router)
app.include_router(uploads.router)
app.include_router(data.router)
app.include_router(test.router)
app.include_router(tags_router.router)
app.include_router(summaries_router.router)
app.include_router(admin_router.router)

@app.get("/", response_model=RootResponse, operation_id="rootIndex")
async def root() -> RootResponse:
    return RootResponse(message="Welcome to the Tornado API")

# ---- 例外ハンドラ ----

@app.exception_handler(AppError)
async def handle_app_error(_request, exc: AppError):
    payload = {"detail": exc.detail}
    headers = exc.headers or None
    return JSONResponse(status_code=exc.status_code, content=payload, headers=headers)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_request, exc: RequestValidationError):
    # FastAPIの標準形式に近い形で返す
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
