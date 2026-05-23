import pytest
from stockbot.llm.base import LLMProvider, LLMResponse, ToolCall
from stockbot.tools.base import Tool
from stockbot.tools.registry import ToolRegistry
from stockbot.memory.store import MemoryStore
from stockbot.memory.history import ConversationHistory
from stockbot.memory.profile import ProfileManager
from stockbot.context import ContextAssembler
from stockbot.quota import QuotaManager
from stockbot.core import AgentCore
from stockbot.events import TextDelta, ToolCallStart, ToolCallEnd, QuotaExceeded


class EchoLLM(LLMProvider):
    """Controllable LLM for testing — can return text or tool_calls."""
    def __init__(self, text: str = "Hello", tool_calls: list[ToolCall] | None = None):
        self.text = text
        self.tool_calls = tool_calls or []

    async def chat(self, messages, tools=None):
        if self.tool_calls:
            return LLMResponse(tool_calls=self.tool_calls, finish_reason="tool_calls")
        return LLMResponse(text=self.text, finish_reason="stop")

    async def chat_stream(self, messages, tools=None):
        for c in self.text:
            yield c


@pytest.fixture
def store(temp_db):
    s = MemoryStore(temp_db)
    s.init_schema()
    return s


@pytest.fixture
def user_id(store):
    return store.create_user("u1", "pw")


@pytest.fixture
def agent(store, user_id):
    llm = EchoLLM(text="你好！有什么可以帮助你的？")
    tools = ToolRegistry()
    async def get_time() -> str:
        return "2026-05-23 14:30:00"
    tools.register(Tool(name="get_time", description="获取当前时间", parameters={
        "type": "object", "properties": {}, "required": []
    }, func=get_time))
    history = ConversationHistory(store)
    profile = ProfileManager(store)
    context = ContextAssembler(tools, profile, history)
    quota = QuotaManager(store, daily_limit=5)
    return AgentCore(llm=llm, tool_registry=tools, context_assembler=context,
                     memory_store=store, history=history, profile=profile,
                     quota=quota, max_turns=8)


class TestAgentCore:
    @pytest.mark.asyncio
    async def test_simple_text_response(self, agent, user_id, store):
        events = []
        async for evt in agent.run(user_id, "你好"):
            events.append(evt)
        assert any(isinstance(e, TextDelta) for e in events)
        texts = [e.content for e in events if isinstance(e, TextDelta)]
        assert "".join(texts) == "你好！有什么可以帮助你的？"

    @pytest.mark.asyncio
    async def test_run_saves_to_history(self, agent, user_id, store):
        async for _ in agent.run(user_id, "你好"):
            pass
        msgs = store.get_history(user_id, 10)
        assert len(msgs) >= 2

    @pytest.mark.asyncio
    async def test_tool_call_flow(self, agent, user_id, store):
        agent.llm.tool_calls = [ToolCall(id="c1", name="get_time", arguments={})]
        agent.llm.text = ""
        events = []
        async for evt in agent.run(user_id, "几点"):
            events.append(evt)
        assert any(isinstance(e, ToolCallStart) and e.name == "get_time" for e in events)
        assert any(isinstance(e, ToolCallEnd) and e.name == "get_time" for e in events)

    @pytest.mark.asyncio
    async def test_quota_exceeded(self, agent, user_id, store):
        for _ in range(5):
            agent.quota.consume(user_id)
        events = []
        async for evt in agent.run(user_id, "你好"):
            events.append(evt)
        assert any(isinstance(e, QuotaExceeded) for e in events)
