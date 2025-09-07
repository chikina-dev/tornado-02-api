"""日付・時刻ユーティリティ（必要最小限）。"""

import datetime
import re
from typing import Optional


def naive_utc_now() -> datetime.datetime:
    """現在時刻（naive UTC 相当）。"""
    return datetime.datetime.now().replace(tzinfo=None)


def as_naive_utc(dt: Optional[datetime.datetime]) -> datetime.datetime:
    """任意の datetime を naive UTC に寄せる（tz情報は除去）。"""
    if dt is None:
        return naive_utc_now()
    if dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None:
        return dt.astimezone().replace(tzinfo=None)
    return dt


_YMD_RE = re.compile(r"^\s*(\d{4})[-/](\d{1,2})[-/](\d{1,2})\s*$")


def parse_ymd_date(date_str: str) -> datetime.date:
    """YYYY-MM-DD（または / 区切り）を datetime.date に変換。"""
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


def day_range(d: datetime.date) -> tuple[datetime.datetime, datetime.datetime]:
    """指定日の 00:00〜23:59:59.999999 を返す。"""
    start = datetime.datetime.combine(d, datetime.time.min)
    end = datetime.datetime.combine(d, datetime.time.max)
    return start, end


def month_range(year: int, month: int) -> tuple[datetime.datetime, datetime.datetime]:
    """指定年月の月初と翌月初を返す。"""
    start = datetime.datetime(year, month, 1)
    if month == 12:
        next_start = datetime.datetime(year + 1, 1, 1)
    else:
        next_start = datetime.datetime(year, month + 1, 1)
    return start, next_start
