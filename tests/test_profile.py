import pytest
from stockbot.memory.store import MemoryStore
from stockbot.memory.profile import ProfileManager


class TestProfileManager:
    @pytest.fixture
    def store(self, temp_db):
        s = MemoryStore(temp_db)
        s.init_schema()
        return s

    @pytest.fixture
    def user_id(self, store):
        return store.create_user("u1", "pw")

    @pytest.fixture
    def profile(self, store):
        return ProfileManager(store)

    def test_add_favorite_stock(self, store, user_id, profile):
        profile.add_favorite(user_id, "600519")
        data = store.get_profile(user_id)
        assert "600519" in data["favorite_stocks"]

    def test_remove_favorite_stock(self, store, user_id, profile):
        profile.add_favorite(user_id, "600519")
        profile.add_favorite(user_id, "000858")
        profile.remove_favorite(user_id, "600519")
        data = store.get_profile(user_id)
        assert data["favorite_stocks"] == ["000858"]

    def test_get_favorites_returns_list(self, store, user_id, profile):
        profile.add_favorite(user_id, "600519")
        favs = profile.get_favorites(user_id)
        assert favs == ["600519"]

    def test_record_query_updates_frequency(self, store, user_id, profile):
        profile.record_query(user_id, "600519")
        profile.record_query(user_id, "600519")
        profile.record_query(user_id, "000858")
        data = store.get_profile(user_id)
        assert data["query_frequency"]["600519"] == 2
        assert data["query_frequency"]["000858"] == 1

    def test_summary_returns_readable_string(self, store, user_id, profile):
        profile.add_favorite(user_id, "600519")
        profile.record_query(user_id, "600519")
        summary = profile.summary(user_id)
        assert "600519" in summary

    def test_no_duplicate_favorites(self, store, user_id, profile):
        profile.add_favorite(user_id, "600519")
        profile.add_favorite(user_id, "600519")
        data = store.get_profile(user_id)
        assert data["favorite_stocks"] == ["600519"]
