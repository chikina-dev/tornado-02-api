from fastapi import FastAPI
from contextlib import asynccontextmanager
from database import database
from routers import users, uploads, data, test, tags as tags_router
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await database.connect()
    try:
        yield
    finally:
        # Shutdown
        await database.disconnect()

app = FastAPI(lifespan=lifespan)

app.include_router(users.router)
app.include_router(uploads.router)
app.include_router(data.router)
app.include_router(test.router)
app.include_router(tags_router.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Tornado API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
