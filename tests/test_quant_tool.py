import pytest
from stockbot.tools.stock_quant import create_quant_tool


class MockQuantPredictor:
    def predict(self, symbol: str) -> dict:
        if symbol == "999999":
            return {"error": "999999 不在量化预测范围内（当前覆盖 CSI300 成分股）"}
        return {
            "symbol": symbol,
            "score": 0.052,
            "rank_pct": 15.0,
            "total": 300,
            "refreshed_at": "2026-05-24",
            "signal": "🟢 强烈看多",
        }


@pytest.fixture
def predictor():
    return MockQuantPredictor()


class TestQuantTool:
    @pytest.mark.asyncio
    async def test_predict_stock_returns_score_and_ranking(self, predictor):
        tool = create_quant_tool(predictor)
        result = await tool.run(symbol="600519")
        assert "600519" in result
        assert "5.20%" in result
        assert "Top 15%" in result
        assert "不构成投资建议" in result

    @pytest.mark.asyncio
    async def test_predict_stock_not_in_universe(self, predictor):
        tool = create_quant_tool(predictor)
        result = await tool.run(symbol="999999")
        assert "失败" in result or "不在" in result

    @pytest.mark.asyncio
    async def test_predict_stock_contains_signal_label(self, predictor):
        tool = create_quant_tool(predictor)
        result = await tool.run(symbol="600519")
        assert any(s in result for s in ["看多", "看空", "中性"])

    def test_quant_predictor_is_available_returns_false(self):
        from stockbot.quant.predictor import QuantPredictor
        assert QuantPredictor.is_available("nonexistent/path") is False
