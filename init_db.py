"""DB 初期化用スクリプト（テスト・ローカル用）。

追加機能:
- 環境変数 MIGRATE_URL_EVAL_TO_DOMAIN=true のとき、url_evaluations を
    ユーザー非依存のドメインキャッシュに移行（user_id 列の削除、url をドメインへ正規化）。
    既存レコードを削除せずに変換・列削除のみを行う。
"""

import asyncio
import os

from database import engine
from sqlalchemy import text
import sqlalchemy
from urllib.parse import urlparse
from models import metadata


def _prepare_sqlite_file_if_needed() -> None:
    """SQLite の場合に DB ファイル用ディレクトリを用意。"""
    url = str(engine.url)
    if url.startswith("sqlite") and ":memory:" not in url:
        path_part = url.split("///", 1)[-1].split("?", 1)[0]
        dir_path = os.path.dirname(path_part)
        if dir_path and dir_path != "." and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)


def _migrate_url_evaluations_to_domain() -> None:
    """Migrate url_evaluations to a global, domain-level cache without user_id.

    Steps:
    - Drop unique constraint on (user_id, url) if present.
    - Normalize url column to domain (netloc) for all rows.
    - Drop foreign key constraint on user_id and drop the user_id column.
    - Add a non-unique index on url for lookup performance.

    This preserves existing rows; it does not delete duplicates across users.
    """
    insp = sqlalchemy.inspect(engine)
    if "url_evaluations" not in insp.get_table_names():
        print("url_evaluations table not found; skipping migration.")
        return

    cols = {c["name"] for c in insp.get_columns("url_evaluations")}
    if "user_id" not in cols:
        print("url_evaluations already migrated (no user_id column).")
        return

    print("Migrating url_evaluations to domain-level cache (dropping user_id)...")
    conn = engine.connect()
    trans = conn.begin()
    try:
        # 1) Drop unique constraint on (user_id, url) if exists
        try:
            for uc in insp.get_unique_constraints("url_evaluations"):
                if set(uc.get("column_names", [])) == {"user_id", "url"} and uc.get("name"):
                    conn.execute(text(f"ALTER TABLE url_evaluations DROP CONSTRAINT {uc['name']}"))
        except Exception as e:
            print(f"Note: could not drop unique constraint cleanly: {e}")

        # 2) Normalize url -> domain for all rows (best-effort)
        rows = conn.execute(text("SELECT id, url FROM url_evaluations")).fetchall()
        for rid, url in rows:
            if not url:
                continue
            # If url is already a bare domain, urlparse(url).netloc will be empty.
            # So fallback to parsing as path and taking the first segment.
            parsed = urlparse(url)
            dom = (parsed.netloc or (parsed.path.split("/", 1)[0] if parsed.path else "")).lower()
            if dom and dom != url:
                conn.execute(text("UPDATE url_evaluations SET url=:u WHERE id=:id"), {"u": dom, "id": rid})

        # 3) Drop FK and user_id column
        try:
            for fk in insp.get_foreign_keys("url_evaluations"):
                if "user_id" in fk.get("constrained_columns", []) and fk.get("name"):
                    conn.execute(text(f"ALTER TABLE url_evaluations DROP CONSTRAINT {fk['name']}"))
        except Exception as e:
            print(f"Note: could not drop FK cleanly: {e}")

        dialect = engine.dialect.name
        if dialect == "postgresql":
            conn.execute(text("ALTER TABLE url_evaluations DROP COLUMN IF EXISTS user_id"))
        else:
            # SQLite 3.35+ supports DROP COLUMN
            try:
                conn.execute(text("ALTER TABLE url_evaluations DROP COLUMN user_id"))
            except Exception as e:
                print(f"Note: DROP COLUMN unsupported; manual migration needed for user_id (dialect={dialect}): {e}")

        # 4) Create non-unique index for faster lookups (best-effort)
        try:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_url_evaluations_url ON url_evaluations (url)"))
        except Exception as e:
            print(f"Note: could not create index on url: {e}")

        trans.commit()
        print("url_evaluations migration complete.")
    except Exception as e:
        trans.rollback()
        print(f"Migration failed: {e}")
    finally:
        conn.close()


def _fix_postgres_sequences() -> None:
    """Repair PostgreSQL sequences for id columns to avoid duplicate key errors.

    Sets each table's id sequence to MAX(id)+1. No-op on non-Postgres.
    Controlled with FIX_PG_SEQUENCES=true.
    """
    if engine.dialect.name != "postgresql":
        print("Sequence fix skipped (not PostgreSQL).")
        return
    tables = [
        "users",
        "files",
        "search_histories",
        "file_summaries",
        "daily_summaries",
        "tags",
        "search_history_tags",
        "daily_summary_tags",
        "file_tags",
        "refresh_tokens",
        "file_upload_webhooks",
        "url_evaluations",
        "user_skill_features",
    ]
    conn = engine.connect()
    try:
        for t in tables:
            try:
                # Use quoted identifiers to handle lowercase table names
                sql = f"""
                SELECT setval(
                    pg_get_serial_sequence('"{t}"', 'id'),
                    COALESCE((SELECT MAX(id) FROM "{t}"), 0) + 1,
                    false
                );
                """
                conn.execute(text(sql))
                print(f"Sequence repaired for {t}")
            except Exception as e:
                print(f"Seq repair skipped for {t}: {e}")
    finally:
        conn.close()


async def main() -> None:
    try:
        _prepare_sqlite_file_if_needed()
        if os.getenv("RESET_DB", "false").lower() in {"1", "true", "yes"}:
            print("Dropping all tables (RESET_DB=true)...")
            from sqlalchemy import MetaData

            md = MetaData()
            md.reflect(bind=engine)
            md.drop_all(bind=engine)
        metadata.create_all(bind=engine)

        # Optional migration: url_evaluations を user_id なし＆ドメイン単位に移行
        if os.getenv("MIGRATE_URL_EVAL_TO_DOMAIN", "false").lower() in {"1", "true", "yes"}:
            _migrate_url_evaluations_to_domain()
        # Optional: Fix PostgreSQL sequences (avoid duplicate key on id)
        if os.getenv("FIX_PG_SEQUENCES", "false").lower() in {"1", "true", "yes"}:
            _fix_postgres_sequences()
        print("Tables ensured.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
