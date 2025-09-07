"""DB 接続の薄いラッパー（databases/SQLAlchemy）。"""

import os

from databases import Database
from dotenv import load_dotenv
from sqlalchemy import MetaData, create_engine

# .env 読み込み（存在する場合のみ）
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("No DATABASE_URL environment variable set")

# 非同期クエリ用
database = Database(DATABASE_URL)

# テーブル作成用（同期）
engine = create_engine(DATABASE_URL)

# メタデータ（テストで create_all に使用）
metadata = MetaData()
