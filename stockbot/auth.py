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

    def list_users(self) -> list[dict]:
        return self.store.list_users()
