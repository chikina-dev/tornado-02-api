import datetime
import re
from typing import Optional


def naive_utc_now() -> datetime.datetime:
    return datetime.datetime.now().replace(tzinfo=None)


def as_naive_utc(dt: Optional[datetime.datetime]) -> datetime.datetime:
    if dt is None:
        return naive_utc_now()
    if dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None:
        return dt.astimezone().replace(tzinfo=None)
    return dt


_YMD_RE = re.compile(r"^\s*(\d{4})[-/](\d{1,2})[-/](\d{1,2})\s*$")


def parse_ymd_date(date_str: str) -> datetime.date:
    if date_str is None:
        raise ValueError("date_str is None")
    s = str(date_str).strip()
    try:
        return datetime.date.fromisoformat(s)
    except Exception:
        pass
    m = _YMD_RE.match(s)
    if not m:
        raise ValueError(f"Invalid date string: {date_str!r}")
    year, month, day = map(int, m.groups())
    return datetime.date(year, month, day)
