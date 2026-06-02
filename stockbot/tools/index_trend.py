from stockbot.tools.base import Tool
from stockbot.index.index_data import IndexDataProvider
from stockbot.index.index_analyzer import IndexAnalyzer


INDEX_NAMES = {
    "000001": "上证指数",
    "399001": "深证成指",
    "399006": "创业板指",
}


def create_index_trend_tool(provider: IndexDataProvider) -> Tool:
    analyzer = IndexAnalyzer()

    async def analyze_index(index_code: str = "000001", period: str = "3m") -> str:
        index_name = INDEX_NAMES.get(index_code, f"指数{index_code}")

        try:
            ohlcv = provider.get_index_history(index_code, period)
        except Exception as e:
            return f"获取{index_name}历史数据失败: {e}"

        try:
            result = analyzer.analyze(ohlcv, index_name=index_name, index_code=index_code)
        except ValueError as e:
            return str(e)

        return analyzer.format_output(result, period)

    return Tool(
        name="analyze_index",
        description=(
            "对大盘指数进行多情景技术分析。输入指数代码(默认000001上证指数)"
            "和周期(1m/3m/6m/1y)，输出乐观/中性/悲观三种情景推演，"
            "包含MA均线、MACD、RSI、量比等指标。支持上证指数、深证成指、"
            "创业板指等主要指数。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "index_code": {
                    "type": "string",
                    "description": "指数代码: 000001(上证指数), 399001(深证成指), 399006(创业板指)",
                    "default": "000001",
                },
                "period": {
                    "type": "string",
                    "description": "分析周期: 1m(1月), 3m(3月,默认), 6m(半年), 1y(1年)",
                    "default": "3m",
                },
            },
            "required": [],
        },
        func=analyze_index,
    )
