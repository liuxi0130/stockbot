from stockbot.memory.store import MemoryStore


class ProfileManager:
    def __init__(self, store: MemoryStore):
        self.store = store

    def _get_data(self, user_id: str) -> dict:
        data = self.store.get_profile(user_id)
        data.setdefault("favorite_stocks", [])
        data.setdefault("query_frequency", {})
        data.setdefault("risk_preference", "未设置")
        return data

    def _save(self, user_id: str, data: dict):
        self.store.set_profile(user_id, data)

    def add_favorite(self, user_id: str, symbol: str):
        data = self._get_data(user_id)
        if symbol not in data["favorite_stocks"]:
            data["favorite_stocks"].append(symbol)
        self._save(user_id, data)

    def remove_favorite(self, user_id: str, symbol: str):
        data = self._get_data(user_id)
        if symbol in data["favorite_stocks"]:
            data["favorite_stocks"].remove(symbol)
        self._save(user_id, data)

    def get_favorites(self, user_id: str) -> list[str]:
        return self._get_data(user_id).get("favorite_stocks", [])

    def record_query(self, user_id: str, symbol: str):
        data = self._get_data(user_id)
        data["query_frequency"][symbol] = data["query_frequency"].get(symbol, 0) + 1
        self._save(user_id, data)

    def summary(self, user_id: str) -> str:
        data = self._get_data(user_id)
        parts = []
        if data.get("favorite_stocks"):
            parts.append(f"关注股票: {', '.join(data['favorite_stocks'])}")
        if data.get("risk_preference"):
            parts.append(f"风险偏好: {data['risk_preference']}")
        if data.get("query_frequency"):
            top = sorted(data["query_frequency"].items(),
                         key=lambda x: x[1], reverse=True)[:3]
            parts.append(f"最近关注: {', '.join(s for s, _ in top)}")
        return " | ".join(parts) if parts else "新用户，暂无画像"
