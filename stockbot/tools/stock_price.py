from stockbot.tools.base import Tool
from stockbot.data.base import DataProvider


def create_price_tool(provider: DataProvider) -> Tool:
    async def get_realtime_quote(symbol: str) -> str:
        quote = provider.get_realtime(symbol)
        arrow = "↑" if quote.change_pct >= 0 else "↓"
        sign = "+" if quote.change_pct >= 0 else ""
        # Determine data freshness
        from datetime import datetime
        data_date = quote.timestamp
        today_str = datetime.now().strftime("%Y-%m-%d")
        freshness = "实时" if data_date == today_str else f"最新交易日 ({data_date})"
        if data_date != today_str:
            freshness += " — 今日数据尚未发布"
        return (
            f"{quote.name} ({quote.symbol})\n"
            f"  最新价: {quote.price:.2f}\n"
            f"  涨跌幅: {sign}{quote.change_pct:.2f}% {arrow}\n"
            f"  成交量: {quote.volume:.0f} 手\n"
            f"  数据时间: {freshness}"
        )

    return Tool(
        name="get_realtime_quote",
        description="获取股票最新交易日行情（T+1 日线数据，非盘中实时）。返回收盘价、涨跌幅、成交量。盘中查询显示上一交易日数据。",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码，如 600519"}
            },
            "required": ["symbol"],
        },
        func=get_realtime_quote,
    )
