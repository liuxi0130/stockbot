from stockbot.tools.base import Tool
from stockbot.data.base import DataProvider


def create_search_tool(provider: DataProvider) -> Tool:
    async def search(query: str) -> str:
        results = provider.search(query)
        if not results:
            return f'未找到与 "{query}" 相关的股票。请尝试其他关键词或完整代码。'
        lines = [f'搜索 "{query}" 的结果:']
        for r in results:
            lines.append(f"  {r['symbol']}  {r['name']}  ({r.get('market', '')})")
        return "\n".join(lines)

    return Tool(
        name="search_stock",
        description="按名称或代码搜索股票。返回匹配的股票代码、名称和市场。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词，如公司名称或股票代码"}
            },
            "required": ["query"],
        },
        func=search,
    )
