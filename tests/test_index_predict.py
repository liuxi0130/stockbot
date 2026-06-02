import pytest
from stockbot.tools.index_predict import create_index_predict_tool


class MockIndexPredictor:
    def __init__(self):
        self.ml_available = False

    def is_ml_available(self) -> bool:
        return self.ml_available

    def predict(self, index_code: str = "000001", period: str = "3m") -> str:
        return (
            f"🔮 上证指数 ({index_code}) 多周期预测 — {period}数据\n"
            "\n"
            "短期信号: 🟢 强烈看多 (85/100)\n"
            "中期信号: 🟡 温和看多 (65/100)\n"
            "\n"
            "⚠️ 仅供参考，不构成投资建议。"
        )


class TestIndexPredictTool:
    @pytest.mark.asyncio
    async def test_predict_contains_scores(self):
        predictor = MockIndexPredictor()
        tool = create_index_predict_tool(predictor)
        result = await tool.run(index_code="000001", period="1m")
        assert "000001" in result
        assert "85" in result or "65" in result

    @pytest.mark.asyncio
    async def test_predict_contains_signal(self):
        predictor = MockIndexPredictor()
        tool = create_index_predict_tool(predictor)
        result = await tool.run()
        assert any(s in result for s in ["看多", "看空", "中性", "震荡"])

    @pytest.mark.asyncio
    async def test_predict_contains_warning(self):
        predictor = MockIndexPredictor()
        tool = create_index_predict_tool(predictor)
        result = await tool.run()
        assert "⚠️" in result or "仅供参考" in result
