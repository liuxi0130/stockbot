import pytest
import bcrypt
from stockbot.memory.store import MemoryStore
from stockbot.auth import AuthManager


class TestAuthManager:
    @pytest.fixture
    def store(self, temp_db):
        s = MemoryStore(temp_db)
        s.init_schema()
        return s

    @pytest.fixture
    def auth(self, store):
        return AuthManager(store, open_registration=True)

    def test_register_creates_user(self, store, auth):
        uid = auth.register("testuser", "password123")
        assert uid is not None
        user = store.get_user("testuser")
        assert user is not None
        assert user["role"] == "user"

    def test_register_duplicate_returns_none(self, store, auth):
        auth.register("testuser", "pw1")
        uid = auth.register("testuser", "pw2")
        assert uid is None

    def test_login_success(self, store, auth):
        auth.register("testuser", "correctpw")
        user = auth.login("testuser", "correctpw")
        assert user is not None
        assert user["username"] == "testuser"

    def test_login_wrong_password(self, store, auth):
        auth.register("testuser", "correctpw")
        user = auth.login("testuser", "wrongpw")
        assert user is None

    def test_login_nonexistent_user(self, store, auth):
        user = auth.login("noone", "pw")
        assert user is None

    def test_register_closed(self, store, auth):
        auth.open_registration = False
        uid = auth.register("newuser", "pw")
        assert uid is None

    def test_password_is_hashed(self, store, auth):
        uid = auth.register("testuser", "mypassword")
        user = store.get_user("testuser")
        assert user["password"] != "mypassword"
        assert bcrypt.checkpw("mypassword".encode(), user["password"].encode())
