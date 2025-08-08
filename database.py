import os
from databases import Database
from sqlalchemy import create_engine, MetaData
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("No DATABASE_URL environment variable set")

# Database instance for query execution
database = Database(DATABASE_URL)

# SQLAlchemy engine for table creation
engine = create_engine(DATABASE_URL)

# Metadata instance to bind to the engine
metadata = MetaData()
