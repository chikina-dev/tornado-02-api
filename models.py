import datetime
import sqlalchemy
from pydantic import BaseModel, EmailStr
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserProfile(BaseModel):
    id: int
    email: EmailStr
    # Add other user data here

class UserProfileActivity(UserProfile):
    month: str  # YYYY-MM
    active_dates: list[int]  # 当月にユーザーのデータが存在する日(1-31)
    file_active_dates: list[int] | None = None  # 種別別が欲しい場合に利用(将来拡張)
    history_active_dates: list[int] | None = None

class Tag(BaseModel):
    id: int
    name: str

class FileUploadWebhook(BaseModel):
    id: int
    user_id: int
    external_id: str  # URL等から取得したID
    created_at: datetime.datetime
    last_executed_at: Optional[datetime.datetime] = None

class SearchHistoryUploadWebhook(BaseModel):
    id: int
    user_id: int
    external_id: str  # URL等から取得したID
    created_at: datetime.datetime
    last_executed_at: Optional[datetime.datetime] = None


metadata = sqlalchemy.MetaData()

users = sqlalchemy.Table(
    "users",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True),
    sqlalchemy.Column("email", sqlalchemy.String, unique=True, index=True),
    sqlalchemy.Column("hashed_password", sqlalchemy.String),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.datetime.utcnow),
)

files = sqlalchemy.Table(
    "files",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id")),
    sqlalchemy.Column("file_path", sqlalchemy.String, unique=True),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.datetime.utcnow),
)

search_histories = sqlalchemy.Table(
    "search_histories",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id")),
    sqlalchemy.Column("query", sqlalchemy.String),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.datetime.utcnow),
)

file_summaries = sqlalchemy.Table(
    "file_summaries",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True),
    sqlalchemy.Column("file_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("files.id")),
    sqlalchemy.Column("summary", sqlalchemy.Text),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.datetime.utcnow),
)

search_history_summaries = sqlalchemy.Table(
    "search_history_summaries",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True),
    sqlalchemy.Column("search_history_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("search_histories.id")),
    sqlalchemy.Column("summary", sqlalchemy.Text),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.datetime.utcnow),
)

tags = sqlalchemy.Table(
    "tags",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id"), index=True),
    sqlalchemy.Column("name", sqlalchemy.String, nullable=False),
    sqlalchemy.UniqueConstraint("user_id", "name", name="uq_user_tag_name"),
)

search_history_tags = sqlalchemy.Table(
    "search_history_tags",
    metadata,
    sqlalchemy.Column("search_history_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("search_histories.id"), primary_key=True),
    sqlalchemy.Column("tag_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("tags.id"), primary_key=True),
)

file_upload_webhooks = sqlalchemy.Table(
    "file_upload_webhooks",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id"), index=True, nullable=False),
    sqlalchemy.Column("external_id", sqlalchemy.String, unique=True, nullable=False),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.datetime.utcnow, nullable=False),
    sqlalchemy.Column("last_executed_at", sqlalchemy.DateTime, nullable=True),
)

search_history_upload_webhooks = sqlalchemy.Table(
    "search_history_upload_webhooks",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id"), index=True, nullable=False),
    sqlalchemy.Column("external_id", sqlalchemy.String, unique=True, nullable=False),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.datetime.utcnow, nullable=False),
    sqlalchemy.Column("last_executed_at", sqlalchemy.DateTime, nullable=True),
)
