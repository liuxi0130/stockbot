from stockbot.memory.store import MemoryStore


class ConversationHistory:
    def __init__(self, store: MemoryStore, history_limit: int = 200):
        self.store = store
        self.history_limit = history_limit

    def get_recent(self, user_id: str, limit: int = 50) -> list[dict]:
        return self.store.get_history(user_id, limit)

    def save_turn(self, user_id: str, user_content: str, assistant_content: str,
                  tool_results: list[dict] | None = None):
        self.store.add_message(user_id, "user", user_content)
        if tool_results:
            for tr in tool_results:
                self.store.add_message(user_id, "tool", str(tr["result"]),
                                       tool_name=tr["name"])
        self.store.add_message(user_id, "assistant", assistant_content)
        self.store.trim_history(user_id, self.history_limit)
