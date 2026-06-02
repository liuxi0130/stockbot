from stockbot.tools.base import Tool
from stockbot.index.index_data import IndexDataProvider


def create_market_overview_tool(provider: IndexDataProvider) -> Tool:
    async def get_market_overview() -> str:
        try:
            quote = provider.get_index_quote("000001")
        except Exception as e:
            return f"获取上证指数行情失败: {e}"

        try:
            breadth = provider.get_market_breadth()
        except Exception:
            breadth = None

        try:
            sectors = provider.get_sector_performance(top_n=5)
        except Exception:
            sectors = []

        arrow = "↑" if quote.change_pct >= 0 else "↓"
        sign = "+" if quote.change_pct >= 0 else ""

        lines = [
            f"\U0001f4c8 大盘概况 — {quote.timestamp}",
            "",
            f"  {quote.name} ({quote.code})",
            f"  最新点位: {quote.price:.2f}",
            f"  涨跌幅: {sign}{quote.change_pct:.2f}% {arrow}",
            f"  涨跌额: {sign}{quote.change_amt:.2f}",
            f"  成交额: {quote.turnover:.0f} 亿元",
        ]

        if breadth:
            total = breadth.up_count + breadth.down_count + breadth.flat_count
            up_pct = breadth.up_count / total * 100 if total > 0 else 0
            down_pct = breadth.down_count / total * 100 if total > 0 else 0
            lines.extend([
                "",
                "\U0001f4ca 市场宽度",
                f"  上涨: {breadth.up_count} 家 ({up_pct:.1f}%)",
                f"  下跌: {breadth.down_count} 家 ({down_pct:.1f}%)",
                f"  平盘: {breadth.flat_count} 家",
                f"  两市成交额: {breadth.total_turnover:.0f} 亿元",
                f"  涨停: {breadth.limit_up} 家  |  跌停: {breadth.limit_down} 家",
            ])

        if sectors:
            lines.append("")
            lines.append("\U0001f525 行业板块")
            top_sectors = [s for s in sectors if s.get("rank") == "top"]
            bottom_sectors = [s for s in sectors if s.get("rank") == "bottom"]
            lines.append("  涨幅居前:")
            for s in top_sectors:
                lines.append(
                    f"    \U0001f7e2 {s['name']}: +{s['change_pct']:.2f}%"
                    + (f"  (领涨: {s.get('leading_stock', '')})" if s.get("leading_stock") else "")
                )
            lines.append("  跌幅居前:")
            for s in bottom_sectors:
                lines.append(
                    f"    \U0001f534 {s['name']}: {s['change_pct']:.2f}%"
                    + (f"  (领跌: {s.get('leading_stock', '')})" if s.get("leading_stock") else "")
                )

        lines.extend([
            "",
            "⚠️ 以上仅为市场概况数据展示，不构成投资建议。",
        ])

        return "\n".join(lines)

    return Tool(
        name="get_market_overview",
        description=(
            "获取A股大盘概况。返回上证指数点位、涨跌幅、市场宽度(涨跌家数)、"
            "两市成交额、行业板块热度等信息。无需参数，适合在市场分析或个股"
            "分析前先了解整体市场环境。"
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        func=get_market_overview,
    )
