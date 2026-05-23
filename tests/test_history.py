import pytest
from stockbot.memory.store import MemoryStore
from stockbot.memory.history import ConversationHistory


class TestConversationHistory:
    @pytest.fixture
    def store(self, temp_db):
        s = MemoryStore(temp_db)
        s.init_schema()
        return s

    @pytest.fixture
    def user_id(self, store):
        return store.create_user("u1", "pw")

    @pytest.fixture
    def history(self, store):
        return ConversationHistory(store, history_limit=200)

    def test_get_recent_returns_messages(self, store, user_id, history):
        store.add_message(user_id, "user", "问题1")
        store.add_message(user_id, "assistant", "回答1")
        msgs = history.get_recent(user_id, 10)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_save_turn_adds_user_and_assistant(self, store, user_id, history):
        history.save_turn(user_id, "用户问题", "助手回答")
        msgs = history.get_recent(user_id, 10)
        assert len(msgs) == 2
        assert msgs[0]["content"] == "用户问题"
        assert msgs[1]["content"] == "助手回答"

    def test_save_turn_with_tool_calls(self, store, user_id, history):
        tool_results = [
            {"name": "get_price", "result": '{"price": 1680}'}
        ]
        history.save_turn(user_id, "茅台价格", "助手回答", tool_results)
        msgs = history.get_recent(user_id, 10)
        assert len(msgs) == 3

    def test_trim_triggered_when_over_limit(self, store, user_id, history):
        history.history_limit = 10
        for i in range(15):
            history.save_turn(user_id, f"msg{i}", f"reply{i}")
        msgs = history.get_recent(user_id, 50)
        assert len(msgs) <= 30

    def test_messages_are_chat_format(self, store, user_id, history):
        history.save_turn(user_id, "你好", "你好！")
        msgs = history.get_recent(user_id, 10)
        for m in msgs:
            assert "role" in m
            assert "content" in m
