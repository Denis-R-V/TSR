from __future__ import annotations
import aiosqlite

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS user_settings (
  user_id INTEGER PRIMARY KEY,
  detector_thr_override REAL NULL,
  classifier_thr_override REAL NULL,
  debug INTEGER NOT NULL DEFAULT 0
);
"""

async def init_db(db_path: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(CREATE_SQL)
        await db.commit()
