import pytest
from stockbot.llm.base import LLMProvider, LLMResponse, ToolCall


class MockLLMProvider(LLMProvider):
    """Fake LLM that returns text or tool_calls based on a canned response."""

    def __init__(self, canned: LLMResponse | None = None):
        self.canned = canned or LLMResponse(text="你好", finish_reason="stop")
        self.last_messages = None
        self.last_tools = None

    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        self.last_messages = messages
        self.last_tools = tools
        return self.canned

    async def chat_stream(self, messages: list[dict], tools: list[dict] | None = None):
        self.last_messages = messages
        self.last_tools = tools
        for char in self.canned.text or "":
            yield char


class TestLLMProviderInterface:
    @pytest.mark.asyncio
    async def test_chat_returns_llm_response(self):
        provider = MockLLMProvider()
        resp = await provider.chat([{"role": "user", "content": "你好"}])
        assert isinstance(resp, LLMResponse)
        assert resp.text == "你好"

    @pytest.mark.asyncio
    async def test_chat_passes_messages_and_tools(self):
        provider = MockLLMProvider()
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        await provider.chat([{"role": "user", "content": "测试"}], tools)
        assert provider.last_messages[0]["content"] == "测试"
        assert len(provider.last_tools) == 1

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self):
        provider = MockLLMProvider(LLMResponse(
            tool_calls=[ToolCall(id="1", name="get_price", arguments={"symbol": "600519"})],
            finish_reason="tool_calls",
        ))
        resp = await provider.chat([{"role": "user", "content": "茅台价格"}])
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "get_price"

    @pytest.mark.asyncio
    async def test_chat_stream_yields_characters(self):
        provider = MockLLMProvider(LLMResponse(text="ABC"))
        chars = []
        async for c in provider.chat_stream([{"role": "user", "content": "hi"}]):
            chars.append(c)
        assert chars == ["A", "B", "C"]
