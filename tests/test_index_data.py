import pytest
from stockbot.index.index_data import (
    IndexQuote, MarketBreadth, IndexDataProvider,
)


class MockIndexProvider(IndexDataProvider):
    """Minimal implementation for testing the abstract interface."""

    def get_index_quote(self, index_code: str = "000001") -> IndexQuote:
        return IndexQuote(
            code=index_code, name="上证指数", price=3350.5,
            change_pct=0.85, change_amt=28.2, volume=1.5e9,
            turnover=3200.0, timestamp="2026-06-02 14:30:00",
        )

    def get_market_breadth(self) -> MarketBreadth:
        return MarketBreadth(
            up_count=2150, down_count=1800, flat_count=350,
            total_turnover=8500.0, limit_up=45, limit_down=12,
        )

    def get_sector_performance(self, top_n: int = 5) -> list[dict]:
        return [
            {"name": "半导体", "change_pct": 3.5, "leading_stock": "中芯国际"},
            {"name": "新能源", "change_pct": 2.8, "leading_stock": "宁德时代"},
            {"name": "房地产", "change_pct": -2.1, "leading_stock": "万科A"},
            {"name": "银行", "change_pct": -1.5, "leading_stock": "工商银行"},
        ]

    def get_index_history(self, index_code: str, period: str) -> list[dict]:
        return [
            {"date": f"2026-05-{20+i:02d}", "open": 3300+i*5, "high": 3320+i*5,
             "low": 3290+i*5, "close": 3310+i*5, "volume": 1e9+i*1e8}
            for i in range(30)
        ]

    def get_index_news(self, limit: int = 5) -> list[dict]:
        return [{"title": "A股三大指数集体收涨", "source": "证券时报",
                 "time": "2026-06-02", "url": ""}]


class TestIndexDataProviderInterface:
    def test_index_quote_fields(self):
        provider = MockIndexProvider()
        q = provider.get_index_quote("000001")
        assert isinstance(q, IndexQuote)
        assert q.code == "000001"
        assert q.name == "上证指数"
        assert q.price > 0
        assert q.turnover > 0

    def test_market_breadth_fields(self):
        provider = MockIndexProvider()
        b = provider.get_market_breadth()
        assert isinstance(b, MarketBreadth)
        assert b.up_count + b.down_count + b.flat_count > 0
        assert b.total_turnover > 0
        assert b.limit_up >= 0

    def test_sector_performance_returns_list(self):
        provider = MockIndexProvider()
        sectors = provider.get_sector_performance(top_n=5)
        assert isinstance(sectors, list)
        assert len(sectors) > 0
        assert "name" in sectors[0]
        assert "change_pct" in sectors[0]

    def test_get_index_history_returns_ohlcv(self):
        provider = MockIndexProvider()
        data = provider.get_index_history("000001", "1m")
        assert isinstance(data, list)
        assert len(data) >= 20
        row = data[0]
        for key in ("date", "open", "high", "low", "close", "volume"):
            assert key in row

    def test_get_index_news_returns_list(self):
        provider = MockIndexProvider()
        news = provider.get_index_news(5)
        assert isinstance(news, list)
        assert len(news) > 0
        assert "title" in news[0]
