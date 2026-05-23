from datetime import date, timedelta
from stockbot.memory.store import MemoryStore
from stockbot.quota import QuotaManager
from stockbot.auth import AuthManager


class AdminService:
    def __init__(self, store: MemoryStore, quota: QuotaManager, auth: AuthManager):
        self.store = store
        self.quota = quota
        self.auth = auth

    def list_users_with_stats(self) -> list[dict]:
        today = date.today().isoformat()
        users = self.store.list_users()
        result = []
        for u in users:
            q = self.store.get_quota(u["id"], today)
            conv_count = len(self.store.get_history(u["id"], 999))
            result.append({
                **u,
                "calls_today": q["calls"],
                "approved_today": q["approved"],
                "limit": u.get("daily_quota", 5) + q["approved"],
                "total_conversations": conv_count,
            })
        return result

    def set_user_quota(self, user_id: str, daily_quota: int):
        self.store.update_user_quota(user_id, daily_quota)

    def approve_user(self, user_id: str, extra: int):
        today = date.today().isoformat()
        self.store.add_approved(user_id, today, extra)

    def get_stats(self) -> dict:
        today = date.today().isoformat()
        users = self.store.list_users()
        total_calls = 0
        for u in users:
            q = self.store.get_quota(u["id"], today)
            total_calls += q["calls"]

        active_users = set()
        for u in users:
            hist = self.store.get_history(u["id"], 1)
            if hist:
                active_users.add(u["id"])

        return {
            "total_users": len(users),
            "active_users_7d": len(active_users),
            "total_calls_today": total_calls,
        }
