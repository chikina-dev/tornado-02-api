"""認証/暗号ユーティリティ（最小限）。"""

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from jose import jwt
from passlib.context import CryptContext

load_dotenv()

# 環境変数から読み込む（本番でハードコード禁止）
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 180

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta if expires_delta else timedelta(minutes=15)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def generate_refresh_token() -> str:
    """リフレッシュトークン（不透明文字列）を生成。"""
    return secrets.token_urlsafe(64)


def hash_token(token: str) -> str:
    """トークンは平文保存しない。SHA-256 でハッシュ化。"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
