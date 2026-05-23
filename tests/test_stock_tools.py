import pytest
from stockbot.data.base import StockQuote, StockHistory
from stockbot.tools.stock_search import create_search_tool
from stockbot.tools.stock_price import create_price_tool
from stockbot.tools.stock_finance import create_finance_tool
from stockbot.tools.stock_trend import create_trend_tool
from stockbot.tools.stock_news import create_news_tool


class MockDataProvider:
    def search(self, query: str) -> list[dict]:
        return [{"symbol": "600519", "name": "贵州茅台", "market": "SH"}]

    def get_realtime(self, symbol: str) -> StockQuote:
        return StockQuote(symbol=symbol, name="贵州茅台", price=1680.0,
                          change_pct=2.3, volume=1e7, timestamp="2026-05-23")

    def get_history(self, symbol: str, period: str) -> StockHistory:
        data = [
            {"date": f"2026-05-{20+i:02d}", "open": 1640 + i * 10,
             "high": 1660 + i * 10, "low": 1630 + i * 10,
             "close": 1650 + i * 10, "volume": 1e7}
            for i in range(5)
        ]
        return StockHistory(symbol=symbol, name="贵州茅台", data=data)

    def get_financial(self, symbol: str) -> dict:
        return {"pe": 25.5, "pb": 8.2, "roe": 32.1, "revenue_growth": 15.3, "eps": 65.8}

    def get_news(self, symbol: str, limit: int = 5) -> list[dict]:
        return [{"title": "茅台涨价", "source": "财经网", "time": "2026-05-23", "url": ""}]


@pytest.fixture
def provider():
    return MockDataProvider()


class TestStockTools:
    @pytest.mark.asyncio
    async def test_search_stock(self, provider):
        tool = create_search_tool(provider)
        result = await tool.run(query="茅台")
        assert "600519" in result
        assert "贵州茅台" in result

    @pytest.mark.asyncio
    async def test_get_realtime_quote(self, provider):
        tool = create_price_tool(provider)
        result = await tool.run(symbol="600519")
        assert "1680" in result

    @pytest.mark.asyncio
    async def test_get_financial_data(self, provider):
        tool = create_finance_tool(provider)
        result = await tool.run(symbol="600519", metric="all")
        assert "25.5" in result or "PE" in result.lower()

    @pytest.mark.asyncio
    async def test_analyze_trend_not_enough_data(self, provider):
        """With only 5 data points, should report insufficient data."""
        tool = create_trend_tool(provider)
        result = await tool.run(symbol="600519", period="1m")
        assert "不足" in result

    @pytest.mark.asyncio
    async def test_search_news(self, provider):
        tool = create_news_tool(provider)
        result = await tool.run(symbol="600519", limit=3)
        assert "茅台涨价" in result


class TestAnalyzeTrendWithEnoughData:
    @pytest.fixture
    def history_provider(self):
        """Provider with 30 data points to satisfy the 20-point minimum."""
        class RichProvider(MockDataProvider):
            def get_history(self, symbol: str, period: str) -> StockHistory:
                data = [
                    {"date": f"2026-05-{i:02d}", "open": 1640 + i * 2,
                     "high": 1660 + i * 2, "low": 1630 + i * 2,
                     "close": 1650 + i * 2, "volume": 1e7}
                    for i in range(1, 31)
                ]
                return StockHistory(symbol=symbol, name="贵州茅台", data=data)
        return RichProvider()

    @pytest.mark.asyncio
    async def test_analyze_trend_scenarios(self, history_provider):
        tool = create_trend_tool(history_provider)
        result = await tool.run(symbol="600519", period="1m")
        assert "乐观" in result
        assert "中性" in result
        assert "悲观" in result
        assert "支撑" in result or "压力" in result
        assert "不构成投资建议" in result
