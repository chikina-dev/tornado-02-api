import os
import asyncio
from database import engine
from models import metadata

# If using a sqlite file DB (e.g. for tests), ensure the file/dir exists before create_all.
# This is optional; for PostgreSQL nothing extra happens.
def _prepare_sqlite_file_if_needed():
    url = str(engine.url)
    # Typical forms: sqlite:///./test.db  OR sqlite:////abs/path/test.db
    if url.startswith("sqlite") and ":memory:" not in url:
        # Extract path after last '///'
        path_part = url.split("///", 1)[-1]
        # Remove query params if any
        path_part = path_part.split("?", 1)[0]
        dir_path = os.path.dirname(path_part)
        if dir_path and dir_path != "." and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        # File will be created automatically by SQLite on first connect; we just ensure folder.

async def main():
    try:
        _prepare_sqlite_file_if_needed()
        metadata.create_all(bind=engine)
        print("Tables created successfully.")
        print("Tables available: ", metadata.tables.keys())
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
