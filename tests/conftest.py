import os
import sys
import pathlib
import uuid
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

TEST_DB_URL = "sqlite+aiosqlite:///./test.db"  # async driver URL for databases
SYNC_DB_URL = "sqlite:///./test.db"            # sync URL for create_all

# Set env BEFORE importing application modules so main/database pick sqlite
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ["DATABASE_URL"] = TEST_DB_URL

from models import metadata  # noqa E402
from database import database  # noqa E402

# Create tables once (synchronous engine)
engine = create_engine(SYNC_DB_URL)
metadata.create_all(bind=engine)

import main as main_module  # noqa E402

@pytest.fixture(scope="session", autouse=True)
async def connect_db():
    await database.connect()
    yield
    await database.disconnect()
    if os.path.exists("test.db"):
        os.remove("test.db")

@pytest.fixture(scope="session")
def app(connect_db):  # FastAPI app is sync accessible
    return main_module.app

@pytest.fixture()
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac

@pytest.fixture()
async def created_user(client):
    email = f"user_{uuid.uuid4().hex[:8]}@example.com"
    payload = {"email": email, "password": "password123"}
    r = await client.post("/create", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    return {"id": data["user_id"], "email": email}

@pytest.fixture()
async def logged_in_client(client, created_user):
    r = await client.post("/login", json={"email": created_user["email"], "password": "password123"})
    assert r.status_code == 200, r.text
    data = r.json()
    token = data.get("access_token")
    assert token, "access_token missing in login response"
    client.headers["Authorization"] = f"Bearer {token}"
    return client
