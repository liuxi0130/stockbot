import pytest
from stockbot.memory.store import MemoryStore
from stockbot.memory.history import ConversationHistory
from stockbot.memory.profile import ProfileManager
from stockbot.tools.base import Tool
from stockbot.tools.registry import ToolRegistry
from stockbot.context import ContextAssembler


class TestContextAssembler:
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
        return ConversationHistory(store)

    @pytest.fixture
    def profile(self, store):
        return ProfileManager(store)

    @pytest.fixture
    def tools(self):
        reg = ToolRegistry()
        async def echo(text: str) -> str:
            return text
        reg.register(Tool(name="echo", description="回显输入", parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }, func=echo))
        return reg

    @pytest.fixture
    def ctx(self, store, history, profile, tools):
        return ContextAssembler(tools, profile, history)

    def test_build_returns_correct_structure(self, ctx, user_id):
        msgs = ctx.build(user_id, "你好")
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "你好"

    def test_system_prompt_contains_tool_descriptions(self, ctx, user_id):
        msgs = ctx.build(user_id, "你好")
        assert "echo" in msgs[0]["content"]

    def test_system_prompt_contains_profile_summary(self, ctx, user_id, profile):
        profile.add_favorite(user_id, "600519")
        msgs = ctx.build(user_id, "你好")
        assert "600519" in msgs[0]["content"]

    def test_system_prompt_contains_warning(self, ctx, user_id):
        msgs = ctx.build(user_id, "你好")
        assert "仅供参考" in msgs[0]["content"]

    def test_build_includes_history(self, ctx, user_id, history):
        history.save_turn(user_id, "问题1", "回答1")
        msgs = ctx.build(user_id, "问题2")
        roles = [m["role"] for m in msgs]
        assert roles.count("user") >= 2
