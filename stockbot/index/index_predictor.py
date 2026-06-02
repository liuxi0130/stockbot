from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from stockbot.index.index_analyzer import IndexAnalyzer

LOGGER = logging.getLogger(__name__)


class IndexSignal(Enum):
    STRONG_BUY = "strong_buy"
    MILD_BUY = "mild_buy"
    NEUTRAL = "neutral"
    MILD_SELL = "mild_sell"
    STRONG_SELL = "strong_sell"


@dataclass
class IndexPrediction:
    short_term_score: int      # 0-100
    mid_term_score: int        # 0-100
    short_term_signal: IndexSignal
    mid_term_signal: IndexSignal
    short_term_detail: str
    mid_term_detail: str
    source: str                # "rule" | "ml" | "combined"


def _score_to_signal(score: int) -> IndexSignal:
    if score >= 80:
        return IndexSignal.STRONG_BUY
    elif score >= 60:
        return IndexSignal.MILD_BUY
    elif score >= 40:
        return IndexSignal.NEUTRAL
    elif score >= 20:
        return IndexSignal.MILD_SELL
    return IndexSignal.STRONG_SELL


class RuleEngine:
    """Deterministic technical-rule-based prediction engine."""

    def __init__(self):
        self.analyzer = IndexAnalyzer()

    def evaluate(self, ohlcv: list[dict]) -> IndexPrediction:
        analysis = self.analyzer.analyze(ohlcv)

        score = 50

        # Trend (0-25 pts)
        if analysis.base_scenario == "bullish":
            score += 20
        elif analysis.base_scenario == "bearish":
            score -= 20

        # MACD (0-15 pts)
        if analysis.macd_bar > 0 and analysis.macd_dif > analysis.macd_dea:
            score += 10
        elif analysis.macd_bar < 0 and analysis.macd_dif < analysis.macd_dea:
            score -= 10

        # RSI (-15 to +15 pts)
        if analysis.overbought:
            score -= 10
        elif analysis.oversold:
            score += 10
        elif analysis.rsi > 55:
            score += 5
        elif analysis.rsi < 45:
            score -= 5

        # Volume (0-10 pts)
        if analysis.vol_ratio > 1.5:
            score += 5
        elif analysis.vol_ratio < 0.5:
            score -= 5

        score = max(0, min(100, score))

        # Mid-term scoring
        mid_score = 50
        if analysis.base_scenario == "bullish":
            mid_score += 25
        elif analysis.base_scenario == "bearish":
            mid_score -= 25
        if analysis.macd_bar > 0:
            mid_score += 10
        else:
            mid_score -= 10
        mid_score = max(0, min(100, mid_score))

        return IndexPrediction(
            short_term_score=score,
            mid_term_score=mid_score,
            short_term_signal=_score_to_signal(score),
            mid_term_signal=_score_to_signal(mid_score),
            short_term_detail=analysis.trend_desc,
            mid_term_detail=(
                f"{analysis.trend_desc}，MACD BAR={analysis.macd_bar:.2f}，"
                f"RSI={analysis.rsi:.0f}"
            ),
            source="rule",
        )


class MLPredictor:
    """Optional LightGBM-based prediction model for index direction."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self._model = None

    def is_available(self) -> bool:
        model_file = self.model_dir / "index_model.pkl"
        return model_file.exists()

    def _load(self):
        import joblib
        model_file = self.model_dir / "index_model.pkl"
        self._model = joblib.load(model_file)

    def _extract_features(self, ohlcv: list[dict]) -> list[float]:
        closes = [d["close"] for d in ohlcv]
        volumes = [d["volume"] for d in ohlcv]
        highs = [d["high"] for d in ohlcv]
        lows = [d["low"] for d in ohlcv]
        n = len(closes)

        ret5 = (closes[-1] / closes[-6] - 1) if n >= 6 else 0.0
        ret10 = (closes[-1] / closes[-11] - 1) if n >= 11 else 0.0
        ret20 = (closes[-1] / closes[-21] - 1) if n >= 21 else 0.0

        import statistics
        vol20 = statistics.stdev(
            [(closes[i] / closes[i-1] - 1) for i in range(max(1, n-20), n)]
        ) if n >= 21 else 0.0

        if n >= 11:
            vol_ma5 = sum(volumes[-5:]) / 5
            vol_ma10 = sum(volumes[-10:]) / 10
            vol_ratio = vol_ma5 / vol_ma10 if vol_ma10 > 0 else 1.0
        else:
            vol_ratio = 1.0

        ma5 = sum(closes[-5:]) / 5 if n >= 5 else closes[-1]
        ma10 = sum(closes[-10:]) / 10 if n >= 10 else closes[-1]
        ma20 = sum(closes[-20:]) / 20 if n >= 20 else closes[-1]
        price_vs_ma20 = (closes[-1] / ma20 - 1) if ma20 > 0 else 0.0

        if n >= 15:
            gains = sum(max(closes[i] - closes[i-1], 0) for i in range(n-14, n))
            losses = sum(max(closes[i-1] - closes[i], 0) for i in range(n-14, n))
            rsi_simple = 100 * gains / (gains + losses) if (gains + losses) > 0 else 50
        else:
            rsi_simple = 50

        range_ratio = (
            (highs[-1] - lows[-1]) / closes[-1] if closes[-1] > 0 else 0.0
        )

        return [
            ret5, ret10, ret20, vol20, vol_ratio,
            price_vs_ma20, rsi_simple / 100, range_ratio,
        ]

    def predict(self, ohlcv: list[dict]) -> Optional[IndexPrediction]:
        if not self.is_available():
            return None
        if len(ohlcv) < 26:
            return None
        try:
            if self._model is None:
                self._load()
            features = self._extract_features(ohlcv)
            short_prob = float(self._model["short_model"].predict_proba([features])[0][1])
            mid_prob = float(self._model["mid_model"].predict_proba([features])[0][1])

            short_score = int(short_prob * 100)
            mid_score = int(mid_prob * 100)

            return IndexPrediction(
                short_term_score=short_score,
                mid_term_score=mid_score,
                short_term_signal=_score_to_signal(short_score),
                mid_term_signal=_score_to_signal(mid_score),
                short_term_detail=f"ML模型预测短期上涨概率 {short_prob:.0%}",
                mid_term_detail=f"ML模型预测中期上涨概率 {mid_prob:.0%}",
                source="ml",
            )
        except Exception as e:
            LOGGER.warning("ML prediction failed: %s", e)
            return None


class IndexPredictor:
    """Multi-horizon index direction predictor."""

    def __init__(self, index_provider, model_dir: str = "data/index_model",
                 ml_enabled: bool = False):
        self.index_provider = index_provider
        self.rule_engine = RuleEngine()
        self.ml_enabled = ml_enabled
        self.ml_predictor = MLPredictor(model_dir) if ml_enabled else None

    def is_ml_available(self) -> bool:
        return self.ml_predictor is not None and self.ml_predictor.is_available()

    def predict(self, index_code: str = "000001", period: str = "3m") -> str:
        try:
            ohlcv = self.index_provider.get_index_history(index_code, period)
        except Exception as e:
            return f"获取指数历史数据失败: {e}"

        if len(ohlcv) < 20:
            return f"历史数据不足（需要至少20个交易日，当前{len(ohlcv)}个），无法进行预测。"

        rule_pred = self.rule_engine.evaluate(ohlcv)
        ml_pred = None
        if self.ml_enabled and self.is_ml_available():
            ml_pred = self.ml_predictor.predict(ohlcv)

        combined = _combine(rule_pred, ml_pred)
        return self._format(index_code, rule_pred, ml_pred, combined, period)

    @staticmethod
    def _score_to_signal(score: int) -> IndexSignal:
        return _score_to_signal(score)

    @staticmethod
    def _signal_to_str(signal: IndexSignal) -> str:
        return {
            IndexSignal.STRONG_BUY: "🟢 强烈看多",
            IndexSignal.MILD_BUY: "🟡 温和看多",
            IndexSignal.NEUTRAL: "⚪ 中性/震荡",
            IndexSignal.MILD_SELL: "🟠 温和看空",
            IndexSignal.STRONG_SELL: "🔴 强烈看空",
        }[signal]

    def _format(self, index_code: str, rule: IndexPrediction,
                ml: Optional[IndexPrediction], combined: IndexPrediction,
                period: str) -> str:
        name_map = {
            "000001": "上证指数", "399001": "深证成指", "399006": "创业板指",
        }
        index_name = name_map.get(index_code, index_code)

        lines = [
            f"🔮 {index_name} ({index_code}) 多周期预测 — {period}数据",
            "",
            "┌─────────────────────────────────────────┐",
            "│ 📊 短期信号 (1-3天)                        │",
            f"│   综合评分: {combined.short_term_score}/100",
            f"│   综合信号: {self._signal_to_str(combined.short_term_signal)}",
        ]

        if ml:
            lines.extend([
                "│                                             │",
                f"│   技术规则: {rule.short_term_score}/100  "
                f"({self._signal_to_str(rule.short_term_signal)})",
                f"│   ML模型:   {ml.short_term_score}/100  "
                f"({self._signal_to_str(ml.short_term_signal)})",
            ])
        else:
            lines.extend([
                "│                                             │",
                f"│   技术规则: {rule.short_term_score}/100",
            ])

        lines.extend([
            "├─────────────────────────────────────────┤",
            "│ 📈 中期信号 (1-4周)                        │",
            f"│   综合评分: {combined.mid_term_score}/100",
            f"│   综合信号: {self._signal_to_str(combined.mid_term_signal)}",
        ])

        if ml:
            lines.extend([
                "│                                             │",
                f"│   技术规则: {rule.mid_term_score}/100  "
                f"({self._signal_to_str(rule.mid_term_signal)})",
                f"│   ML模型:   {ml.mid_term_score}/100  "
                f"({self._signal_to_str(ml.mid_term_signal)})",
            ])

        if not ml:
            lines.append("│   (ML增强模型未加载，仅使用技术规则信号)")

        lines.extend([
            "└─────────────────────────────────────────┘",
            "",
            "⚠️ 以上为多因子综合研判结果，基于历史数据统计规律，",
            "不构成投资建议。指数走势受宏观经济、货币政策、国际形势等",
            "多重不可量化因素影响，预测存在不确定性。",
        ])

        return "\n".join(lines)


def _combine(rule: IndexPrediction, ml: Optional[IndexPrediction] = None) -> IndexPrediction:
    if ml is None:
        return rule

    short_score = int(rule.short_term_score * 0.5 + ml.short_term_score * 0.5)
    mid_score = int(rule.mid_term_score * 0.5 + ml.mid_term_score * 0.5)

    return IndexPrediction(
        short_term_score=short_score,
        mid_term_score=mid_score,
        short_term_signal=_score_to_signal(short_score),
        mid_term_signal=_score_to_signal(mid_score),
        short_term_detail=f"规则({rule.short_term_score}) + ML({ml.short_term_score})",
        mid_term_detail=f"规则({rule.mid_term_score}) + ML({ml.mid_term_score})",
        source="combined",
    )
