import pytest
from stockbot.memory.store import MemoryStore
from stockbot.quota import QuotaManager


class TestQuotaManager:
    @pytest.fixture
    def store(self, temp_db):
        s = MemoryStore(temp_db)
        s.init_schema()
        return s

    @pytest.fixture
    def user_id(self, store):
        return store.create_user("u1", "pw")

    @pytest.fixture
    def manager(self, store):
        return QuotaManager(store, daily_limit=5)

    def test_check_allows_within_limit(self, store, user_id, manager):
        result = manager.check(user_id)
        assert not result.blocked
        assert result.remain == 5
        assert result.used == 0

    def test_consume_increments_count(self, store, user_id, manager):
        manager.consume(user_id)
        manager.consume(user_id)
        result = manager.check(user_id)
        assert result.used == 2
        assert result.remain == 3
        assert not result.blocked

    def test_check_blocks_when_exhausted(self, store, user_id, manager):
        for _ in range(5):
            manager.consume(user_id)
        result = manager.check(user_id)
        assert result.blocked
        assert result.remain == 0

    def test_admin_approve_adds_extra(self, store, user_id, manager):
        for _ in range(5):
            manager.consume(user_id)
        manager.approve(user_id, 5)
        result = manager.check(user_id)
        assert not result.blocked
        assert result.remain == 5
        assert result.limit == 10
