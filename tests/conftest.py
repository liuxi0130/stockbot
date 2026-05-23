import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        Path(path).unlink()
    except (PermissionError, OSError):
        pass
    for ext in ["-wal", "-shm"]:
        try:
            Path(path + ext).unlink()
        except (PermissionError, OSError):
            pass


@pytest.fixture
def sample_config():
    return {
        "llm": {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "api_key": "test-key",
            "max_tokens": 4096,
            "temperature": 0.3,
        },
        "quota": {"daily_limit": 5, "admin_password": "admin123"},
        "data": {"providers": ["akshare"]},
        "memory": {"history_limit": 200, "db_path": ":memory:"},
        "auth": {"open_registration": True},
    }
