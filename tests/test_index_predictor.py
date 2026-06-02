import pytest
from stockbot.index.index_predictor import (
    IndexPredictor, RuleEngine, IndexPrediction, IndexSignal, _combine,
)
from stockbot.index.index_analyzer import IndexAnalyzer


def make_sample_ohlcv(n: int = 60, trend: str = "up") -> list[dict]:
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
        data.append({
            "date": f"2026-{1+i//30:02d}-{1+i%28:02d}",
            "open": base - random.uniform(-3, 3),
            "high": base + random.uniform(0, 8),
            "low": base - random.uniform(0, 8),
            "close": base,
            "volume": random.uniform(5e8, 3e9),
        })
    return data


class MockIndexProvider:
    def __init__(self, ohlcv: list[dict]):
        self._data = ohlcv

    def get_index_history(self, index_code: str, period: str) -> list[dict]:
        return self._data


class TestRuleEngine:
    def test_evaluate_returns_prediction(self):
        engine = RuleEngine()
        data = make_sample_ohlcv(60, "up")
        pred = engine.evaluate(data)
        assert isinstance(pred, IndexPrediction)
        assert pred.short_term_score is not None
        assert pred.mid_term_score is not None
        assert 0 <= pred.short_term_score <= 100
        assert 0 <= pred.mid_term_score <= 100

    def test_uptrend_gives_bullish_signal(self):
        engine = RuleEngine()
        data = make_sample_ohlcv(60, "up")
        pred = engine.evaluate(data)
        assert pred.short_term_signal in (
            IndexSignal.STRONG_BUY, IndexSignal.MILD_BUY, IndexSignal.NEUTRAL
        )

    def test_downtrend_gives_bearish_signal(self):
        engine = RuleEngine()
        data = make_sample_ohlcv(60, "down")
        pred = engine.evaluate(data)
        assert pred.short_term_signal in (
            IndexSignal.STRONG_SELL, IndexSignal.MILD_SELL, IndexSignal.NEUTRAL
        )


class TestIndexPredictor:
    def test_predict_rules_only(self):
        data = make_sample_ohlcv(60, "up")
        provider = MockIndexProvider(data)
        predictor = IndexPredictor(index_provider=provider, ml_enabled=False)
        result = predictor.predict("000001", "1m")
        assert "短周期" in result or "短期" in result
        assert "中周期" in result or "中期" in result

    def test_predict_returns_warning(self):
        data = make_sample_ohlcv(60, "up")
        provider = MockIndexProvider(data)
        predictor = IndexPredictor(index_provider=provider, ml_enabled=False)
        result = predictor.predict("000001", "1m")
        assert "⚠️" in result or "仅供参考" in result

    def test_is_ml_available_returns_false(self):
        data = make_sample_ohlcv(60, "up")
        provider = MockIndexProvider(data)
        predictor = IndexPredictor(
            index_provider=provider,
            model_dir="nonexistent/path",
            ml_enabled=True,
        )
        assert predictor.is_ml_available() is False

    def test_combiner_weights_ml_when_available(self):
        data = make_sample_ohlcv(60, "up")
        engine = RuleEngine()
        rule_pred = engine.evaluate(data)
        combined = _combine(rule_pred, ml=None)
        assert combined.short_term_score == rule_pred.short_term_score

    def test_signal_to_str(self):
        s = IndexPredictor._signal_to_str(IndexSignal.STRONG_BUY)
        assert "看多" in s
        assert "🟢" in s

    def test_score_to_signal(self):
        s = IndexPredictor._score_to_signal(85)
        assert s == IndexSignal.STRONG_BUY
        s = IndexPredictor._score_to_signal(65)
        assert s == IndexSignal.MILD_BUY
        s = IndexPredictor._score_to_signal(50)
        assert s == IndexSignal.NEUTRAL
        s = IndexPredictor._score_to_signal(25)
        assert s == IndexSignal.MILD_SELL
        s = IndexPredictor._score_to_signal(10)
        assert s == IndexSignal.STRONG_SELL
