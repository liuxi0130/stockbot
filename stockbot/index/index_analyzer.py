from dataclasses import dataclass
from typing import Optional


@dataclass
class IndexAnalysisResult:
    index_code: str
    index_name: str
    last_price: float
    ma5: float
    ma10: float
    ma20: float
    ma60: Optional[float]
    macd_dif: float
    macd_dea: float
    macd_bar: float
    rsi: float
    vol_ratio: float
    base_scenario: str          # 'bullish' | 'bearish' | 'neutral'
    trend_desc: str
    support_level: Optional[float]
    resistance_level: Optional[float]
    overbought: bool
    oversold: bool


class IndexAnalyzer:
    """Pure-rule technical analysis for stock indices.

    All calculations are deterministic and Python-only — no LLM
    involvement in numeric computation.
    """

    MIN_DATA_POINTS = 20

    # ── public ──────────────────────────────────────────────

    def analyze(self, ohlcv: list[dict], index_name: str = "上证指数",
                index_code: str = "000001") -> IndexAnalysisResult:
        """Run full technical analysis on index OHLCV data."""
        if len(ohlcv) < self.MIN_DATA_POINTS:
            raise ValueError(
                f"数据不足: 需要至少 {self.MIN_DATA_POINTS} 个交易日，"
                f"当前仅 {len(ohlcv)} 个"
            )

        closes = [d["close"] for d in ohlcv]
        volumes = [d["volume"] for d in ohlcv]

        ma5 = self._calc_ma(closes, 5)
        ma10 = self._calc_ma(closes, 10)
        ma20 = self._calc_ma(closes, 20)
        ma60 = self._calc_ma(closes, 60) if len(closes) >= 60 else [None] * len(closes)

        dif, dea, macd_hist = self._calc_macd(closes)
        rsi = self._calc_rsi(closes, 14)
        vol_ma5 = self._calc_ma(volumes, 5)

        last_close = closes[-1]
        ma5_val = ma5[-1] or last_close
        ma10_val = ma10[-1] or last_close
        ma20_val = ma20[-1] or last_close
        ma60_val = ma60[-1]
        rsi_val = rsi[-1] if rsi[-1] is not None else 50
        macd_dif_val = dif[-1] or 0
        macd_dea_val = dea[-1] or 0
        macd_bar_val = macd_hist[-1] or 0
        vol_ratio = volumes[-1] / (vol_ma5[-1] or 1) if vol_ma5[-1] else 1.0

        # Trend classification
        if ma5_val > ma10_val > ma20_val:
            trend_desc = "多头排列，短期上涨趋势"
            base_scenario = "bullish"
        elif ma5_val < ma10_val < ma20_val:
            trend_desc = "空头排列，短期下跌趋势"
            base_scenario = "bearish"
        else:
            trend_desc = "均线交织，处于震荡整理"
            base_scenario = "neutral"

        support, resistance = self._find_levels(closes)

        return IndexAnalysisResult(
            index_code=index_code,
            index_name=index_name,
            last_price=last_close,
            ma5=ma5_val,
            ma10=ma10_val,
            ma20=ma20_val,
            ma60=ma60_val,
            macd_dif=macd_dif_val,
            macd_dea=macd_dea_val,
            macd_bar=macd_bar_val,
            rsi=rsi_val,
            vol_ratio=vol_ratio,
            base_scenario=base_scenario,
            trend_desc=trend_desc,
            support_level=support,
            resistance_level=resistance,
            overbought=(rsi_val > 70),
            oversold=(rsi_val < 30),
        )

    def format_output(self, result: IndexAnalysisResult, period: str = "3m") -> str:
        """Format analysis result as readable string for LLM consumption."""
        p = result
        rsi_note = "(超买)" if p.overbought else ("(超卖)" if p.oversold else "(中性)")
        vol_note = "(放量)" if p.vol_ratio > 1.5 else ("(缩量)" if p.vol_ratio < 0.5 else "(正常)")

        # Build scenario conditions
        if p.base_scenario == "bullish":
            bullish_cond = ["当前多头趋势延续", "成交量维持或放大"]
            bullish_target = f"{p.last_price * 1.03:.0f}-{p.last_price * 1.08:.0f}"
            bearish_cond = [f"跌破 20 日均线 ({p.ma20:.0f})", "MACD 死叉"]
            bearish_target = f"{p.support_level or p.last_price * 0.95:.0f}"
            neutral_target = f"{p.last_price * 0.98:.0f}-{p.last_price * 1.02:.0f}"
        elif p.base_scenario == "bearish":
            bearish_cond = ["当前空头趋势延续", "成交量持续萎缩"]
            bearish_target = f"{p.support_level or p.last_price * 0.92:.0f}"
            bullish_cond = [f"放量突破 10 日均线 ({p.ma10:.0f})", "MACD 金叉"]
            bullish_target = f"{p.last_price * 1.02:.0f}-{p.last_price * 1.06:.0f}"
            neutral_target = f"{p.last_price * 0.98:.0f}-{p.last_price * 1.02:.0f}"
        else:
            bullish_cond = ["放量突破震荡区间上沿"]
            bullish_target = f"{p.resistance_level or p.last_price * 1.05:.0f}-{p.last_price * 1.10:.0f}"
            bearish_cond = ["放量跌破震荡区间下沿"]
            bearish_target = f"{p.support_level or p.last_price * 0.90:.0f}"
            neutral_target = f"{p.support_level or p.last_price * 0.95:.0f}-{p.resistance_level or p.last_price * 1.05:.0f}"

        sup_parts = []
        res_parts = []
        if p.support_level:
            sup_parts.append(f"{p.support_level:.0f}")
        if p.ma20:
            sup_parts.append(f"MA20 {p.ma20:.0f}")
        if p.ma60:
            sup_parts.append(f"MA60 {p.ma60:.0f}")
        if p.resistance_level:
            res_parts.append(f"{p.resistance_level:.0f}")
        if p.ma10:
            res_parts.append(f"MA10 {p.ma10:.0f}")
        if p.ma5:
            res_parts.append(f"MA5 {p.ma5:.0f}")

        return (
            f"📊 {p.index_name} ({p.index_code}) 技术面分析 — {period}周期\n\n"
            f"当前点位: {p.last_price:.2f}\n"
            f"当前状态: {p.trend_desc}\n"
            f"MACD: DIF={p.macd_dif:.2f} DEA={p.macd_dea:.2f} BAR={p.macd_bar:.2f}\n"
            f"RSI(14): {p.rsi:.1f} {rsi_note}\n"
            f"量比: {p.vol_ratio:.1f} {vol_note}\n\n"
            f"┌──────────────────────────────────────────────────┐\n"
            f"│ 🟢 乐观情景                                        │\n"
            f"│   条件: {'、'.join(bullish_cond)}"
            f"\n│   目标: {bullish_target}\n"
            f"├──────────────────────────────────────────────────┤\n"
            f"│ 🟡 中性情景                                        │\n"
            f"│   区间: {neutral_target}\n"
            f"├──────────────────────────────────────────────────┤\n"
            f"│ 🔴 悲观情景                                        │\n"
            f"│   条件: {'、'.join(bearish_cond)}"
            f"\n│   目标: {bearish_target}\n"
            f"└──────────────────────────────────────────────────┘\n\n"
            f"支撑: {', '.join(sup_parts) if sup_parts else 'N/A'}    "
            f"压力: {', '.join(res_parts) if res_parts else 'N/A'}\n\n"
            f"⚠️ 以上为基于技术指标的多情景推演，不构成投资建议。"
            f"指数走势受宏观经济、政策变化、国际形势等不可量化因素影响。"
        )

    # ── indicator calculations ─────────────────────────────

    @staticmethod
    def _calc_ma(values: list[float], window: int) -> list[Optional[float]]:
        result = []
        for i in range(len(values)):
            if i < window - 1:
                result.append(None)
            else:
                result.append(sum(values[i - window + 1:i + 1]) / window)
        return result

    @staticmethod
    def _calc_ema(values: list[float], window: int) -> list[Optional[float]]:
        result = []
        multiplier = 2 / (window + 1)
        for i in range(len(values)):
            if i == 0:
                result.append(values[0])
            elif i < window - 1:
                result.append(None)
            elif i == window - 1:
                result.append(sum(values[:window]) / window)
            else:
                result.append(
                    (values[i] - result[i - 1]) * multiplier + result[i - 1]
                )
        return result

    def _calc_macd(self, closes: list[float]) -> tuple:
        ema12 = self._calc_ema(closes, 12)
        ema26 = self._calc_ema(closes, 26)
        dif = [None if e12 is None or e26 is None else e12 - e26
               for e12, e26 in zip(ema12, ema26)]
        dea = self._calc_ema([d if d is not None else 0 for d in dif], 9)
        macd_hist = [None if d is None or dea_i is None else 2 * (d - dea_i)
                     for d, dea_i in zip(dif, dea)]
        return dif, dea, macd_hist

    @staticmethod
    def _calc_rsi(closes: list[float], window: int = 14) -> list[Optional[float]]:
        result = [None] * len(closes)
        gains = []
        losses = []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i - 1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        for i in range(len(gains)):
            if i < window - 1:
                continue
            avg_gain = sum(gains[i - window + 1:i + 1]) / window
            avg_loss = sum(losses[i - window + 1:i + 1]) / window
            if avg_loss == 0:
                result[i + 1] = 100.0
            else:
                result[i + 1] = 100 - 100 / (1 + avg_gain / avg_loss)
        return result

    @staticmethod
    def _find_levels(closes: list[float]) -> tuple:
        """Find nearest support/resistance from historical price levels."""
        last = closes[-1]
        sorted_levels = sorted(set(round(c, 0) for c in closes), reverse=True)
        resistance = None
        support = None
        for lvl in sorted_levels:
            if lvl > last and (resistance is None or lvl < resistance):
                resistance = lvl
            if lvl < last and (support is None or lvl > support):
                support = lvl
        return support, resistance
