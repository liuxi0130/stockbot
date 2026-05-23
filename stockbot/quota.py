import os
from dataclasses import dataclass
from datetime import date
from stockbot.memory.store import MemoryStore


@dataclass
class QuotaResult:
    used: int
    limit: int
    remain: int
    blocked: bool


class QuotaManager:
    def __init__(self, store: MemoryStore, daily_limit: int = 5):
        self.store = store
        self.daily_limit = daily_limit
        self.admin_password_hash = os.environ.get("ADMIN_PASS", "admin123")

    def check(self, user_id: str) -> QuotaResult:
        today = date.today().isoformat()
        row = self.store.get_quota(user_id, today)
        used = row["calls"]
        approved = row["approved"]
        limit = self.daily_limit + approved
        remain = limit - used
        return QuotaResult(used=used, limit=limit, remain=max(0, remain),
                           blocked=(remain <= 0))

    def consume(self, user_id: str):
        today = date.today().isoformat()
        self.store.incr_quota(user_id, today)

    def approve(self, user_id: str, extra: int):
        today = date.today().isoformat()
        self.store.add_approved(user_id, today, extra)

    def reset(self, user_id: str):
        today = date.today().isoformat()
        self.store.add_approved(user_id, today, 0)

    def verify_password(self, password: str) -> bool:
        return password == self.admin_password_hash
