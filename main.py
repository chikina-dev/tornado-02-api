from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from database import database
from routers import users, uploads, data, test, tags as tags_router
from routers import summaries as summaries_router
from routers import admin as admin_router
from models import RootResponse
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo
import datetime
from scheduler.summaries import regenerate_for_all_users
from scheduler.tokens import prune_expired_refresh_tokens

@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    scheduler = AsyncIOScheduler(timezone=ZoneInfo("Asia/Tokyo"))

    async def daily_job():
        today = datetime.date.today()
        target_date = today - datetime.timedelta(days=1)
        await regenerate_for_all_users(target_date)

    scheduler.add_job(daily_job, CronTrigger(hour=0, minute=0))
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
        "https://chikina-dev.github.io"
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

@app.get("/", response_model=RootResponse)
async def root():
    return {"message": "Welcome to the Tornado API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
