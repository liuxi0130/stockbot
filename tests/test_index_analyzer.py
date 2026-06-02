import pytest
from stockbot.index.index_analyzer import IndexAnalyzer, IndexAnalysisResult


def make_sample_ohlcv(n: int = 60, trend: str = "up") -> list[dict]:
    """Generate synthetic OHLCV data for testing."""
    import random
    random.seed(42)
    base = 3300.0
    data = []
    for i in range(n):
        if trend == "up":
            base += 2.0 + random.uniform(-5, 8)
        elif trend == "down":
            base -= 2.0 + random.uniform(-5, 8)
        else:
            base += random.uniform(-8, 8)
        close = base
        data.append({
            "date": f"2026-{1+i//30:02d}-{1+i%28:02d}",
            "open": close - random.uniform(-3, 3),
            "high": close + random.uniform(0, 8),
            "low": close - random.uniform(0, 8),
            "close": close,
            "volume": random.uniform(5e8, 3e9),
        })
    return data


class TestIndexAnalyzer:
    def test_analyze_returns_result_dataclass(self):
        analyzer = IndexAnalyzer()
        data = make_sample_ohlcv(60, "up")
        result = analyzer.analyze(data, index_name="上证指数", index_code="000001")
        assert isinstance(result, IndexAnalysisResult)
        assert result.index_code == "000001"
        assert result.trend_desc != ""

    def test_upward_trend_detected(self):
        analyzer = IndexAnalyzer()
        data = make_sample_ohlcv(60, "up")
        result = analyzer.analyze(data, index_name="上证指数", index_code="000001")
        assert result.base_scenario in ("bullish",)
        assert result.ma5 > result.ma10

    def test_downward_trend_detected(self):
        analyzer = IndexAnalyzer()
        data = make_sample_ohlcv(60, "down")
        result = analyzer.analyze(data, index_name="上证指数", index_code="000001")
        assert result.base_scenario == "bearish"
        assert result.rsi < 60

    def test_insufficient_data_raises(self):
        analyzer = IndexAnalyzer()
        data = make_sample_ohlcv(10, "up")
        with pytest.raises(ValueError, match="需要至少"):
            analyzer.analyze(data, index_name="上证指数", index_code="000001")

    def test_result_contains_support_resistance(self):
        analyzer = IndexAnalyzer()
        data = make_sample_ohlcv(60, "neutral")
        result = analyzer.analyze(data, index_name="上证指数", index_code="000001")
        assert result.support_level is not None
        assert result.resistance_level is not None
        assert result.support_level < result.resistance_level

    def test_macd_rsi_are_computed(self):
        analyzer = IndexAnalyzer()
        data = make_sample_ohlcv(60, "up")
        result = analyzer.analyze(data, index_name="上证指数", index_code="000001")
        assert result.macd_dif is not None
        assert result.macd_dea is not None
        assert 0 <= result.rsi <= 100

    def test_format_output_returns_string_with_scenarios(self):
        analyzer = IndexAnalyzer()
        data = make_sample_ohlcv(60, "up")
        result = analyzer.analyze(data, index_name="上证指数", index_code="000001")
        output = analyzer.format_output(result, period="3m")
        assert "上证指数" in output
        assert "乐观" in output
        assert "中性" in output
        assert "悲观" in output
        assert "⚠️" in output or "仅供参考" in output or "投资建议" in output
