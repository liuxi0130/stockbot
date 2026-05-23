from stockbot.tools.base import Tool
from stockbot.data.base import DataProvider


SUPPORTED_METRICS = ["pe", "pb", "roe", "revenue_growth", "eps", "all"]


def create_finance_tool(provider: DataProvider) -> Tool:
    async def get_financial_data(symbol: str, metric: str = "all") -> str:
        if metric not in SUPPORTED_METRICS:
            return f'不支持的指标: {metric}。可用指标: {", ".join(SUPPORTED_METRICS)}'

        data = provider.get_financial(symbol)
        if not data:
            return f"未找到 {symbol} 的财务数据。"

        labels = {
            "pe": "市盈率(PE)", "pb": "市净率(PB)",
            "roe": "净资产收益率(ROE)%", "revenue_growth": "营收增长%",
            "eps": "每股收益(EPS)",
        }

        lines = [f"📊 {symbol} 财务指标:"]
        if metric == "all":
            for key, label in labels.items():
                val = data.get(key)
                if val is not None:
                    lines.append(f"  {label}: {val}")
        else:
            val = data.get(metric)
            label = labels.get(metric, metric)
            lines.append(f"  {label}: {val if val is not None else '无数据'}")
        return "\n".join(lines)

    return Tool(
        name="get_financial_data",
        description="获取股票财务指标。支持 PE、PB、ROE、营收增长率、EPS。用 'all' 获取全部。",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码"},
                "metric": {"type": "string",
                           "description": f"指标: {', '.join(SUPPORTED_METRICS)}",
                           "default": "all"},
            },
            "required": ["symbol"],
        },
        func=get_financial_data,
    )
