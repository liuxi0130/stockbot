from stockbot.tools.base import Tool
from stockbot.data.base import DataProvider


def create_news_tool(provider: DataProvider) -> Tool:
    async def search_news(symbol: str, limit: int = 5) -> str:
        news = provider.get_news(symbol, limit)
        if not news:
            return f"未找到 {symbol} 的相关新闻。"
        lines = [f"📰 {symbol} 相关新闻:"]
        for i, n in enumerate(news, 1):
            src = n.get("source", "")
            t = n.get("time", "")
            lines.append(f"  {i}. {n['title']}")
            lines.append(f"     来源: {src}  {t}")
        return "\n".join(lines)

    return Tool(
        name="search_news",
        description="搜索股票相关新闻。返回最近新闻标题、来源和时间。",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码"},
                "limit": {"type": "integer", "description": "返回条数，默认 5", "default": 5},
            },
            "required": ["symbol"],
        },
        func=search_news,
    )
