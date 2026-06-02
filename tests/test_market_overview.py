import pytest
from stockbot.index.index_data import IndexQuote, MarketBreadth
from stockbot.tools.market_overview import create_market_overview_tool


class MockIndexProvider:
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
            {"name": "半导体", "change_pct": 3.5, "leading_stock": "中芯国际", "rank": "top"},
            {"name": "新能源", "change_pct": 2.8, "leading_stock": "宁德时代", "rank": "top"},
            {"name": "房地产", "change_pct": -2.1, "leading_stock": "万科A", "rank": "bottom"},
            {"name": "银行", "change_pct": -1.5, "leading_stock": "工商银行", "rank": "bottom"},
        ]


class TestMarketOverviewTool:
    @pytest.mark.asyncio
    async def test_overview_contains_index_price(self):
        provider = MockIndexProvider()
        tool = create_market_overview_tool(provider)
        result = await tool.run()
        assert "3350.5" in result
        assert "上证指数" in result

    @pytest.mark.asyncio
    async def test_overview_contains_breadth(self):
        provider = MockIndexProvider()
        tool = create_market_overview_tool(provider)
        result = await tool.run()
        assert "2150" in result
        assert "8500" in result or "成交额" in result

    @pytest.mark.asyncio
    async def test_overview_contains_sectors(self):
        provider = MockIndexProvider()
        tool = create_market_overview_tool(provider)
        result = await tool.run()
        assert "半导体" in result
        assert "银行" in result

    @pytest.mark.asyncio
    async def test_overview_contains_warning(self):
        provider = MockIndexProvider()
        tool = create_market_overview_tool(provider)
        result = await tool.run()
        assert "仅供参考" in result or "⚠️" in result
