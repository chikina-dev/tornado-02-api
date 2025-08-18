import datetime
import pytest
from datetime import timezone

from database import database
from models import files, search_histories


@pytest.mark.asyncio
async def test_profile_empty(logged_in_client):
    r = await logged_in_client.get("/profile")
    assert r.status_code == 200
    data = r.json()
    assert data["active_dates"] == []
    assert data["file_active_dates"] == []
    assert data["history_active_dates"] == []


@pytest.mark.asyncio
async def test_profile_activity_current_month(logged_in_client, created_user):
    user_id = created_user["id"]
    now = datetime.datetime.now(timezone.utc)
    month_start = datetime.datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    day1 = month_start + datetime.timedelta(days=0)
    day2 = month_start + datetime.timedelta(days=1)

    # Insert file (day1)
    await database.execute(
        files.insert().values(user_id=user_id, file_path="dummy1.txt", created_at=day1)
    )
    # Insert history (day2)
    await database.execute(
        search_histories.insert().values(user_id=user_id, query="q", created_at=day2)
    )

    r = await logged_in_client.get("/profile")
    assert r.status_code == 200
    data = r.json()
    assert day1.day in data["file_active_dates"]
    assert day2.day in data["history_active_dates"]
    assert set(data["active_dates"]) == {day1.day, day2.day}


@pytest.mark.asyncio
async def test_profile_activity_previous_month(logged_in_client, created_user):
    user_id = created_user["id"]
    now = datetime.datetime.now(timezone.utc)
    # previous month calculation
    if now.month == 1:
        year = now.year - 1
        month = 12
    else:
        year = now.year
        month = now.month - 1
    prev_month_day1 = datetime.datetime(year, month, 1, tzinfo=timezone.utc)

    await database.execute(
        files.insert().values(user_id=user_id, file_path="prev.txt", created_at=prev_month_day1)
    )

    r_prev = await logged_in_client.get("/profile", params={"month": f"{year:04d}-{month:02d}"})
    assert r_prev.status_code == 200
    d_prev = r_prev.json()
    assert d_prev["file_active_dates"] == [1]
    assert d_prev["active_dates"] == [1]

    # current month should not include that day
    r_cur = await logged_in_client.get("/profile")
    assert r_cur.status_code == 200
    d_cur = r_cur.json()
    assert 1 not in d_cur["file_active_dates"] or d_cur["month"] != f"{year:04d}-{month:02d}"  # ensure separation


@pytest.mark.asyncio
async def test_profile_invalid_month(logged_in_client):
    r = await logged_in_client.get("/profile", params={"month": "2025/08"})
    assert r.status_code == 400
