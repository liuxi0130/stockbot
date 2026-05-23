import sqlite3
import json
import uuid


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    username    TEXT UNIQUE NOT NULL,
    password    TEXT NOT NULL,
    role        TEXT DEFAULT 'user',
    daily_quota INTEGER DEFAULT 5,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id),
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    tool_name   TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS profile (
    user_id     TEXT PRIMARY KEY REFERENCES users(id),
    value       TEXT NOT NULL DEFAULT '{}',
    updated_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS quota (
    user_id     TEXT NOT NULL REFERENCES users(id),
    date        TEXT NOT NULL,
    calls       INTEGER DEFAULT 0,
    approved    INTEGER DEFAULT 0,
    PRIMARY KEY (user_id, date)
);

CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_conv_user_date ON conversations(user_id, date(created_at));
"""


class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _fetch_all(self, sql: str, params: tuple = ()):
        with self._connect() as conn:
            return conn.execute(sql, params).fetchall()

    def _fetch_one(self, sql: str, params: tuple = ()):
        with self._connect() as conn:
            return conn.execute(sql, params).fetchone()

    def _execute(self, sql: str, params: tuple = ()):
        with self._connect() as conn:
            conn.execute(sql, params)

    def init_schema(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # ── Users ──

    def create_user(self, username: str, password_hash: str, role: str = "user") -> str:
        uid = str(uuid.uuid4())
        self._execute(
            "INSERT INTO users (id, username, password, role) VALUES (?, ?, ?, ?)",
            (uid, username, password_hash, role),
        )
        self._execute("INSERT INTO profile (user_id) VALUES (?)", (uid,))
        return uid

    def get_user(self, username: str) -> dict | None:
        row = self._fetch_one("SELECT * FROM users WHERE username = ?", (username,))
        return dict(row) if row else None

    def get_user_by_id(self, user_id: str) -> dict | None:
        row = self._fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))
        return dict(row) if row else None

    def list_users(self) -> list[dict]:
        rows = self._fetch_all("SELECT * FROM users ORDER BY created_at DESC")
        return [dict(r) for r in rows]

    def update_user_quota(self, user_id: str, daily_quota: int):
        self._execute("UPDATE users SET daily_quota = ? WHERE id = ?", (daily_quota, user_id))

    # ── Conversations ──

    def add_message(self, user_id: str, role: str, content: str, tool_name: str | None = None) -> str:
        mid = str(uuid.uuid4())
        self._execute(
            "INSERT INTO conversations (id, user_id, role, content, tool_name) VALUES (?, ?, ?, ?, ?)",
            (mid, user_id, role, content, tool_name),
        )
        return mid

    def get_history(self, user_id: str, limit: int = 50) -> list[dict]:
        rows = self._fetch_all(
            "SELECT role, content, tool_name, created_at FROM conversations "
            "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
        return [dict(r) for r in reversed(rows)]

    def trim_history(self, user_id: str, keep: int = 200):
        self._execute(
            "DELETE FROM conversations WHERE id NOT IN ("
            "SELECT id FROM conversations WHERE user_id = ? ORDER BY created_at DESC LIMIT ?"
            ") AND user_id = ?",
            (user_id, keep, user_id),
        )

    # ── Profile ──

    def get_profile(self, user_id: str) -> dict:
        row = self._fetch_one("SELECT value FROM profile WHERE user_id = ?", (user_id,))
        if row and row["value"]:
            return json.loads(row["value"])
        return {}

    def set_profile(self, user_id: str, data: dict):
        self._execute(
            "INSERT INTO profile (user_id, value) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET value = ?, updated_at = datetime('now')",
            (user_id, json.dumps(data, ensure_ascii=False),
             json.dumps(data, ensure_ascii=False)),
        )

    # ── Quota ──

    def get_quota(self, user_id: str, dt: str) -> dict:
        row = self._fetch_one(
            "SELECT calls, approved FROM quota WHERE user_id = ? AND date = ?",
            (user_id, dt),
        )
        if row:
            return {"calls": row["calls"], "approved": row["approved"]}
        return {"calls": 0, "approved": 0}

    def incr_quota(self, user_id: str, dt: str):
        self._execute(
            "INSERT INTO quota (user_id, date, calls) VALUES (?, ?, 1) "
            "ON CONFLICT(user_id, date) DO UPDATE SET calls = calls + 1",
            (user_id, dt),
        )

    def add_approved(self, user_id: str, dt: str, n: int):
        self._execute(
            "INSERT INTO quota (user_id, date, approved) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id, date) DO UPDATE SET approved = approved + ?",
            (user_id, dt, n, n),
        )
