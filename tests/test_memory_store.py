import pytest
from stockbot.memory.store import MemoryStore


class TestMemoryStore:
    @pytest.fixture
    def store(self, temp_db):
        s = MemoryStore(temp_db)
        s.init_schema()
        return s

    def test_init_schema_creates_tables(self, store):
        tables = store._fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = [t[0] for t in tables]
        assert "users" in table_names
        assert "conversations" in table_names
        assert "profile" in table_names
        assert "quota" in table_names

    def test_create_and_get_user(self, store):
        uid = store.create_user("testuser", "hashed_pw", "user")
        user = store.get_user("testuser")
        assert user["username"] == "testuser"
        assert user["role"] == "user"
        assert user["daily_quota"] == 50

    def test_get_user_returns_none_for_missing(self, store):
        assert store.get_user("noone") is None

    def test_add_and_get_messages(self, store):
        uid = store.create_user("u1", "pw", "user")
        store.add_message(uid, "user", "你好")
        store.add_message(uid, "assistant", "你好！")
        history = store.get_history(uid, 10)
        assert len(history) == 2
        assert history[1]["role"] == "assistant"
        assert history[0]["role"] == "user"

    def test_message_stores_tool_name(self, store):
        uid = store.create_user("u1", "pw", "user")
        store.add_message(uid, "tool", '{"price": 1680}', tool_name="get_realtime_quote")
        history = store.get_history(uid, 10)
        assert history[0]["tool_name"] == "get_realtime_quote"

    def test_profile_crud(self, store):
        uid = store.create_user("u1", "pw", "user")
        store.set_profile(uid, {"favorite_stocks": ["600519"]})
        profile = store.get_profile(uid)
        assert profile["favorite_stocks"] == ["600519"]

    def test_get_profile_returns_empty_dict_for_new_user(self, store):
        uid = store.create_user("u1", "pw", "user")
        assert store.get_profile(uid) == {}

    def test_quota_tracking(self, store):
        uid = store.create_user("u1", "pw", "user")
        q = store.get_quota(uid, "2026-05-23")
        assert q["calls"] == 0
        assert q["approved"] == 0

        store.incr_quota(uid, "2026-05-23")
        q = store.get_quota(uid, "2026-05-23")
        assert q["calls"] == 1

        store.incr_quota(uid, "2026-05-23")
        q = store.get_quota(uid, "2026-05-23")
        assert q["calls"] == 2

    def test_add_approved_quota(self, store):
        uid = store.create_user("u1", "pw", "user")
        store.add_approved(uid, "2026-05-23", 10)
        q = store.get_quota(uid, "2026-05-23")
        assert q["approved"] == 10

    def test_trim_history(self, store):
        uid = store.create_user("u1", "pw", "user")
        for i in range(10):
            store.add_message(uid, "user", f"msg {i}")
        store.trim_history(uid, keep=5)
        history = store.get_history(uid, 50)
        assert len(history) == 5

    def test_list_users(self, store):
        store.create_user("alice", "pw", "user")
        store.create_user("bob", "pw", "user")
        users = store.list_users()
        assert len(users) == 2
        usernames = {u["username"] for u in users}
        assert usernames == {"alice", "bob"}

    def test_update_user_quota(self, store):
        uid = store.create_user("u1", "pw", "user")
        store.update_user_quota(uid, 20)
        user = store.get_user_by_id(uid)
        assert user["daily_quota"] == 20

    def test_create_and_get_valid_session(self, store):
        uid = store.create_user("sess_user", "pw", "user")
        token = "test-token-123"
        expires = "2099-12-31 23:59:59"
        store.create_session(uid, token, expires)
        session = store.get_session(token)
        assert session is not None
        assert session["user_id"] == uid
        assert session["token"] == token

    def test_expired_session_returns_none(self, store):
        uid = store.create_user("exp_user", "pw", "user")
        token = "expired-token"
        expires = "2020-01-01 00:00:00"
        store.create_session(uid, token, expires)
        session = store.get_session(token)
        assert session is None

    def test_delete_session(self, store):
        uid = store.create_user("del_user", "pw", "user")
        token = "del-token"
        expires = "2099-12-31 23:59:59"
        store.create_session(uid, token, expires)
        store.delete_session(token)
        assert store.get_session(token) is None

    def test_cleanup_expired_sessions(self, store):
        uid = store.create_user("cln_user", "pw", "user")
        store.create_session(uid, "good", "2099-12-31 23:59:59")
        store.create_session(uid, "bad", "2020-01-01 00:00:00")
        store.cleanup_expired_sessions()
        assert store.get_session("good") is not None
        assert store.get_session("bad") is None
