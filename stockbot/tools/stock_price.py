from stockbot.tools.base import Tool
from stockbot.data.base import DataProvider


def create_price_tool(provider: DataProvider) -> Tool:
    async def get_realtime_quote(symbol: str) -> str:
        quote = provider.get_realtime(symbol)
        arrow = "↑" if quote.change_pct >= 0 else "↓"
        sign = "+" if quote.change_pct >= 0 else ""
        return (
            f"{quote.name} ({quote.symbol})\n"
            f"  最新价: {quote.price:.2f}\n"
            f"  涨跌幅: {sign}{quote.change_pct:.2f}% {arrow}\n"
            f"  成交量: {quote.volume:.0f} 手\n"
            f"  时间: {quote.timestamp}"
        )

    return Tool(
        name="get_realtime_quote",
        description="获取股票实时行情。返回最新价格、涨跌幅、成交量等信息。",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码，如 600519"}
            },
            "required": ["symbol"],
        },
        func=get_realtime_quote,
    )
