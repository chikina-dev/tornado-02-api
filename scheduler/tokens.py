from database import database
from models import refresh_tokens
from utils.datetime_utils import naive_utc_now


async def prune_expired_refresh_tokens():
    now = naive_utc_now()
    await database.execute(
        refresh_tokens.delete().where(refresh_tokens.c.expires_at < now)
    )
