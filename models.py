"""PydanticモデルとSQLAlchemyテーブル定義（最小限）。"""

import datetime
from typing import List, Optional, Any, Dict

import sqlalchemy
from pydantic import BaseModel, EmailStr, RootModel

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserProfile(BaseModel):
    id: int
    email: EmailStr

class UserProfileActivity(UserProfile):
    month: str
    active_dates: list[int]
    file_active_dates: list[int] | None = None
    history_active_dates: list[int] | None = None

class WebhooksInfo(BaseModel):
    file_external_id: str

class CreateAccountResponse(BaseModel):
    message: str
    user_id: int
    email: EmailStr
    webhooks: WebhooksInfo

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    refresh_token: str | None = None

class RefreshRequest(BaseModel):
    refresh_token: str

class RefreshResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    refresh_token: str | None = None

class UploadFileResponse(BaseModel):
    file_id: int
    filename: str

class UploadHistoryResponse(BaseModel):
    history_id: int

class WebhookUploadFileResponse(BaseModel):
    file_id: int
    saved: bool

class FilesListResponse(BaseModel):
    date: str  # YYYY-MM-DD
    file_ids: List[int]

class FileBinaryResponse(BaseModel):
    file_id: int
    filename: str
    content_base64: str  # Base64 文字列
    content_type: Optional[str] = None
    created_at: Optional[datetime.datetime] = None

class HistoryItem(BaseModel):
    id: int
    url: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[datetime.datetime] = None

class HistoryListResponse(BaseModel):
    date: str  # YYYY-MM-DD
    histories: List[HistoryItem]

class FileWebhookLogItem(BaseModel):
    id: int
    external_id: str
    created_at: datetime.datetime

class LogResponse(BaseModel):
    date: str  # YYYY-MM-DD
    file_webhooks: Optional[List[FileWebhookLogItem]] = None

class ArchiveFileSummaryItem(BaseModel):
    id: int
    file_id: int
    created_at: datetime.datetime

class ArchiveHistorySummaryItem(BaseModel):
    id: int
    search_history_id: int
    created_at: datetime.datetime

class ArchiveResponse(BaseModel):
    date: str  # YYYY-MM-DD
    file_summaries: Optional[List[ArchiveFileSummaryItem]] = None
    history_summaries: Optional[List[ArchiveHistorySummaryItem]] = None

class RootResponse(BaseModel):
    message: str

class TestEndpointResponse(BaseModel):
    test: str
    date: Optional[str] = None
    tags: List[str]

class DailySummaryResponse(BaseModel):
    date: str  # YYYY-MM-DD
    markdown: str
    created_at: datetime.datetime
    tags: List[str]

class AnyDictModel(RootModel[Dict[str, Any]]):
    """任意のJSON辞書を包む薄い型。"""
    pass

class Tag(BaseModel):
    id: int
    name: str

class FileUploadWebhook(BaseModel):
    id: int
    user_id: int
    external_id: str  # 外部ID（URLなど）
    created_at: datetime.datetime
    last_executed_at: Optional[datetime.datetime] = None

class HistoryPayload(BaseModel):
    url: str | None = None
    title: str | None = None
    description: str | None = None
    timestamp: datetime.datetime | None = None
    external_id: str | None = None

class AdminRegenerateResponse(BaseModel):
    user_id: int
    date: str
    file_summaries: int
    history_summaries: int
    daily: bool
    tags: int
    links: Optional[int] = None
    # Optional extras when triggering URL evaluation and skill analysis
    url_evaluated_new: Optional[int] = None
    skills_upserted: Optional[int] = None
    feedback: Optional[str] = None

class AnalysisResponse(BaseModel):
    user_id: int
    date: str
    feedback: str
    url_count: int
    avg_difficulty: Dict[str, float | None]
    top_categories: List[str]

class EvaluateUrlsResponse(BaseModel):
    evaluated: int
    users: int
    by_user: Dict[int, int]

class EpsonPrintJobResponse(BaseModel):
    job: Dict[str, Any]
    printed: bool

class FeedbackResponse(BaseModel):
    user_id: int
    date: str
    feedback: str

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
    sqlalchemy.Column("url", sqlalchemy.String, nullable=True),
    sqlalchemy.Column("title", sqlalchemy.String, nullable=True),
    sqlalchemy.Column("description", sqlalchemy.Text, nullable=True),
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

daily_summaries = sqlalchemy.Table(
    "daily_summaries",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id"), index=True, nullable=False),
    sqlalchemy.Column("date", sqlalchemy.Date, nullable=False),
    sqlalchemy.Column("markdown", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.datetime.utcnow, nullable=False),
    sqlalchemy.UniqueConstraint("user_id", "date", name="uq_user_date_daily_summary"),
)

# 1日単位の分析フィードバック本文（任意保存）
daily_feedback = sqlalchemy.Table(
    "daily_feedback",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id"), index=True, nullable=False),
    sqlalchemy.Column("date", sqlalchemy.Date, nullable=False),
    sqlalchemy.Column("feedback", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.datetime.utcnow, nullable=False),
    sqlalchemy.UniqueConstraint("user_id", "date", name="uq_user_date_daily_feedback"),
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
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True),
    sqlalchemy.Column(
        "search_history_id",
        sqlalchemy.Integer,
        sqlalchemy.ForeignKey("search_histories.id"),
        index=True,
        nullable=False,
    ),
    sqlalchemy.Column("tag_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("tags.id"), index=True, nullable=False),
    sqlalchemy.UniqueConstraint("search_history_id", "tag_id", name="uq_search_history_tag"),
)

daily_summary_tags = sqlalchemy.Table(
    "daily_summary_tags",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True),
    sqlalchemy.Column(
        "daily_summary_id",
        sqlalchemy.Integer,
        sqlalchemy.ForeignKey("daily_summaries.id"),
        index=True,
        nullable=False,
    ),
    sqlalchemy.Column("tag_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("tags.id"), index=True, nullable=False),
    sqlalchemy.UniqueConstraint("daily_summary_id", "tag_id", name="uq_daily_summary_tag"),
)

file_tags = sqlalchemy.Table(
    "file_tags",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True),
    sqlalchemy.Column(
        "file_id",
        sqlalchemy.Integer,
        sqlalchemy.ForeignKey("files.id"),
        index=True,
        nullable=False,
    ),
    sqlalchemy.Column("tag_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("tags.id"), index=True, nullable=False),
    sqlalchemy.UniqueConstraint("file_id", "tag_id", name="uq_file_tag"),
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

refresh_tokens = sqlalchemy.Table(
    "refresh_tokens",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id"), index=True, nullable=False),
    sqlalchemy.Column("token_hash", sqlalchemy.String, unique=True, nullable=False),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.datetime.utcnow, nullable=False),
    sqlalchemy.Column("expires_at", sqlalchemy.DateTime, nullable=False),
    sqlalchemy.Column("revoked", sqlalchemy.Boolean, default=False, nullable=False),
)

# ドメイン単位のLLM評価キャッシュ（再呼び出し抑制）
url_evaluations = sqlalchemy.Table(
    "url_evaluations",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True),
    # `url` 列にはドメイン文字列を格納（例: "example.com"）
    sqlalchemy.Column("url", sqlalchemy.String, nullable=False, unique=True),
    sqlalchemy.Column("difficulty_score", sqlalchemy.Float, nullable=True),
    sqlalchemy.Column("specialization_level", sqlalchemy.String, nullable=True),
    sqlalchemy.Column("skill_categories", sqlalchemy.JSON, nullable=True),
    sqlalchemy.Column("evaluated_at", sqlalchemy.DateTime, default=datetime.datetime.utcnow, nullable=False),
)

# ユーザー別スキル特徴量の集計（表示/LLM入力用）
user_skill_features = sqlalchemy.Table(
    "user_skill_features",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id"), index=True, nullable=False),
    sqlalchemy.Column("skill", sqlalchemy.String, index=True, nullable=False),
    sqlalchemy.Column("heuristic_score_sum", sqlalchemy.Float, nullable=True),
    sqlalchemy.Column("mean_difficulty", sqlalchemy.Float, nullable=True),
    sqlalchemy.Column("n_pages", sqlalchemy.Integer, nullable=True),
    sqlalchemy.Column("mean_engagement", sqlalchemy.Float, nullable=True),
    sqlalchemy.Column("updated_at", sqlalchemy.DateTime, default=datetime.datetime.utcnow, nullable=False),
    sqlalchemy.UniqueConstraint("user_id", "skill", name="uq_user_skill_feature"),
)
