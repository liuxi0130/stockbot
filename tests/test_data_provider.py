import pytest
from stockbot.data.base import DataProvider, StockQuote, StockHistory


class MockDataProvider(DataProvider):
    """Minimal implementation for testing abstract interface."""

    def search(self, query: str) -> list[dict]:
        return [{"symbol": "600519", "name": "贵州茅台", "market": "SH"}]

    def get_realtime(self, symbol: str) -> StockQuote:
        return StockQuote(
            symbol=symbol, name="贵州茅台", price=1680.0,
            change_pct=2.3, volume=12345678, timestamp="2026-05-23 14:00:00"
        )

    def get_history(self, symbol: str, period: str) -> StockHistory:
        return StockHistory(symbol=symbol, name="贵州茅台", data=[
            {"date": "2026-05-20", "open": 1640, "high": 1685, "low": 1635,
             "close": 1680, "volume": 10000000}
        ])

    def get_financial(self, symbol: str) -> dict:
        return {"pe": 25.5, "pb": 8.2, "roe": 32.1, "revenue_growth": 15.3}

    def get_news(self, symbol: str, limit: int) -> list[dict]:
        return [{"title": "测试新闻", "source": "测试来源", "time": "2026-05-23", "url": ""}]


class TestDataProviderInterface:
    def test_can_instantiate_mock_provider(self):
        provider = MockDataProvider()
        assert isinstance(provider, DataProvider)

    def test_search_returns_list_of_dicts(self):
        provider = MockDataProvider()
        results = provider.search("茅台")
        assert len(results) > 0
        assert "symbol" in results[0]
        assert "name" in results[0]

    def test_get_realtime_returns_stock_quote(self):
        provider = MockDataProvider()
        quote = provider.get_realtime("600519")
        assert isinstance(quote, StockQuote)
        assert quote.price > 0

    def test_get_history_returns_stock_history(self):
        provider = MockDataProvider()
        history = provider.get_history("600519", "1m")
        assert isinstance(history, StockHistory)
        assert len(history.data) > 0

    def test_get_financial_returns_dict(self):
        provider = MockDataProvider()
        fin = provider.get_financial("600519")
        assert "pe" in fin

    def test_get_news_returns_list(self):
        provider = MockDataProvider()
        news = provider.get_news("600519", 5)
        assert isinstance(news, list)
        assert len(news) > 0
