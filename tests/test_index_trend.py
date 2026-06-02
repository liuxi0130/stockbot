import pytest
from stockbot.index.index_data import IndexQuote, MarketBreadth
from stockbot.tools.index_trend import create_index_trend_tool
import random


class MockIndexProvider:
    def get_index_quote(self, index_code: str = "000001") -> IndexQuote:
        return IndexQuote(
            code=index_code, name="上证指数", price=3350.5,
            change_pct=0.85, change_amt=28.2, volume=1.5e9,
            turnover=3200.0, timestamp="2026-06-02",
        )

    def get_index_history(self, index_code: str, period: str) -> list[dict]:
        random.seed(24)
        base = 3300.0
        data = []
        for i in range(60):
            base += 2.0 + random.uniform(-5, 8)
            data.append({
                "date": f"2026-{1+i//30:02d}-{1+i%28:02d}",
                "open": base - random.uniform(0, 3),
                "high": base + random.uniform(0, 8),
                "low": base - random.uniform(0, 8),
                "close": base,
                "volume": random.uniform(1e9, 5e9),
            })
        return data


class TestIndexTrendTool:
    @pytest.mark.asyncio
    async def test_trend_contains_index_name(self):
        provider = MockIndexProvider()
        tool = create_index_trend_tool(provider)
        result = await tool.run(index_code="000001", period="1m")
        assert "上证指数" in result

    @pytest.mark.asyncio
    async def test_trend_contains_scenarios(self):
        provider = MockIndexProvider()
        tool = create_index_trend_tool(provider)
        result = await tool.run(index_code="000001", period="1m")
        assert "乐观" in result and "中性" in result and "悲观" in result

    @pytest.mark.asyncio
    async def test_trend_contains_warning(self):
        provider = MockIndexProvider()
        tool = create_index_trend_tool(provider)
        result = await tool.run(index_code="000001", period="1m")
        assert "⚠️" in result or "仅供参考" in result or "投资建议" in result

    @pytest.mark.asyncio
    async def test_trend_insufficient_data(self):
        provider = MockIndexProvider()
        provider.get_index_history = lambda *a, **kw: [
            {"date": f"2026-05-{20+i:02d}", "open": 3300, "high": 3320,
             "low": 3290, "close": 3310, "volume": 1e9}
            for i in range(5)
        ]
        tool = create_index_trend_tool(provider)
        result = await tool.run(index_code="000001", period="1m")
        assert "不足" in result or "至少" in result
