import os
import asyncio
from database import engine
from models import metadata

# テストDBの初期化用
def _prepare_sqlite_file_if_needed():
    url = str(engine.url)
    if url.startswith("sqlite") and ":memory:" not in url:
        path_part = url.split("///", 1)[-1]
        path_part = path_part.split("?", 1)[0]
        dir_path = os.path.dirname(path_part)
        if dir_path and dir_path != "." and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

async def main():
    try:
        _prepare_sqlite_file_if_needed()
        if os.getenv("RESET_DB", "false").lower() in {"1", "true", "yes"}:
            print("Dropping all tables (RESET_DB=true)...")
            from sqlalchemy import MetaData
            md = MetaData()
            md.reflect(bind=engine)
            md.drop_all(bind=engine)
        metadata.create_all(bind=engine)
        print("Tables ensured.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
