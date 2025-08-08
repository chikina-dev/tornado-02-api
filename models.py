import datetime
import sqlalchemy
from pydantic import BaseModel, EmailStr

# Pydantic models for API data validation
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

# SQLAlchemy metadata
metadata = sqlalchemy.MetaData()

# Database table definitions
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
