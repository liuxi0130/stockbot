import asyncio
from stockbot.tools.base import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get_schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def describe(self) -> str:
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)

    async def execute(self, name: str, args: dict) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f'错误: 未找到工具 "{name}"。可用工具: {", ".join(self._tools.keys())}'
        try:
            return await asyncio.wait_for(tool.run(**args), timeout=10.0)
        except asyncio.TimeoutError:
            return f'错误: 工具 "{name}" 执行超时'
        except Exception as e:
            return f'错误: 工具 "{name}" 执行失败: {e}'
