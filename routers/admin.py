import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.datetime_utils import parse_ymd_date
from scheduler.summaries import regenerate_for_user_date


class RegenerateRequest(BaseModel):
    password: str
    user_id: int
    date: str  # YYYY-MM-DD


router = APIRouter(tags=["admin"])


@router.post("/admin/regenerate")
async def admin_regenerate(body: RegenerateRequest):
    admin_pass = os.getenv("ADMIN_PASS")
    if not admin_pass or body.password != admin_pass:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        target_date = parse_ymd_date(body.date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    result = await regenerate_for_user_date(body.user_id, target_date)
    return {"user_id": body.user_id, "date": target_date.isoformat(), **result}
