import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
from stockbot.memory.store import MemoryStore


class AuthManager:
    def __init__(self, store: MemoryStore, open_registration: bool = True):
        self.store = store
        self.open_registration = open_registration

    def register(self, username: str, password: str) -> str | None:
        if not self.open_registration:
            return None
        if self.store.get_user(username):
            return None
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        return self.store.create_user(username, hashed, role="user")

    def login(self, username: str, password: str) -> dict | None:
        user = self.store.get_user(username)
        if not user:
            return None
        if not bcrypt.checkpw(password.encode(), user["password"].encode()):
            return None
        return user

    def create_session(self, user_id: str) -> str:
        """Create a persistent session token (30-day expiry), store in DB, return token."""
        self.store.cleanup_expired_sessions()
        token = str(uuid.uuid4())
        expires_at = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
        self.store.create_session(user_id, token, expires_at)
        return token

    def validate_session(self, token: str) -> dict | None:
        """Validate a session token, return user dict or None if expired/invalid."""
        session = self.store.get_session(token)
        if not session:
            return None
        user = self.store.get_user_by_id(session["user_id"])
        return user

    def delete_session(self, token: str):
        self.store.delete_session(token)

    def list_users(self) -> list[dict]:
        return self.store.list_users()
