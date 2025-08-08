from fastapi import FastAPI
from database import database
from routers import users, uploads, data, test
import uvicorn

app = FastAPI()

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

app.include_router(users.router)
app.include_router(uploads.router)
app.include_router(data.router)
app.include_router(test.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Tornado API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
