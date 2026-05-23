import pytest
from stockbot.tools.base import Tool
from stockbot.tools.registry import ToolRegistry


async def echo_func(text: str) -> str:
    return f"echo: {text}"


async def add_func(a: int, b: int) -> str:
    return str(a + b)


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(Tool(name="echo", description="Echo back input", parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text"}},
        "required": ["text"],
    }, func=echo_func))
    reg.register(Tool(name="add", description="Add two numbers", parameters={
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "First"},
            "b": {"type": "integer", "description": "Second"},
        },
        "required": ["a", "b"],
    }, func=add_func))
    return reg


class TestTool:
    def test_tool_to_openai_schema(self):
        tool = Tool(name="echo", description="Echo", parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }, func=echo_func)
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"

    @pytest.mark.asyncio
    async def test_tool_run(self):
        tool = Tool(name="echo", description="Echo", parameters={}, func=echo_func)
        result = await tool.run(text="hello")
        assert result == "echo: hello"


class TestToolRegistry:
    def test_get_schemas_returns_all_tools(self, registry):
        schemas = registry.get_schemas()
        assert len(schemas) == 2

    def test_describe_returns_tool_list(self, registry):
        desc = registry.describe()
        assert "echo" in desc
        assert "add" in desc

    @pytest.mark.asyncio
    async def test_execute_runs_tool(self, registry):
        result = await registry.execute("echo", {"text": "hi"})
        assert result == "echo: hi"

    @pytest.mark.asyncio
    async def test_execute_returns_error_for_unknown_tool(self, registry):
        result = await registry.execute("unknown", {})
        assert "错误" in result

    @pytest.mark.asyncio
    async def test_execute_handles_exception(self, registry):
        async def failing(x: str) -> str:
            raise RuntimeError("test failure")
        registry.register(Tool(name="fail", description="x", parameters={
            "type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]
        }, func=failing))
        result = await registry.execute("fail", {"x": "y"})
        assert "错误" in result
