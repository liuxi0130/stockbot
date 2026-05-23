from stockbot.tools.base import Tool
from stockbot.data.base import DataProvider


def _calc_ma(closes: list[float], window: int) -> list[float | None]:
    result = []
    for i in range(len(closes)):
        if i < window - 1:
            result.append(None)
        else:
            result.append(sum(closes[i - window + 1:i + 1]) / window)
    return result


def _calc_ema(closes: list[float], window: int) -> list[float | None]:
    result = []
    multiplier = 2 / (window + 1)
    for i in range(len(closes)):
        if i == 0:
            result.append(closes[0])
        elif i < window - 1:
            result.append(None)
        elif i == window - 1:
            result.append(sum(closes[:window]) / window)
        else:
            result.append((closes[i] - result[i - 1]) * multiplier + result[i - 1])
    return result


def _calc_macd(closes: list[float]) -> tuple:
    ema12 = _calc_ema(closes, 12)
    ema26 = _calc_ema(closes, 26)
    dif = [None if e12 is None or e26 is None else e12 - e26
           for e12, e26 in zip(ema12, ema26)]
    dea = _calc_ema([d if d is not None else 0 for d in dif], 9)
    macd_hist = [None if d is None or dea_i is None else 2 * (d - dea_i)
                 for d, dea_i in zip(dif, dea)]
    return dif, dea, macd_hist


def _calc_rsi(closes: list[float], window: int = 14) -> list[float | None]:
    result = [None] * len(closes)
    gains, losses = [], []
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
            result[i + 1] = 100
        else:
            result[i + 1] = 100 - 100 / (1 + avg_gain / avg_loss)
    return result


def _find_levels(closes: list[float]):
    high = max(closes)
    low = min(closes)
    mid = (high + low) / 2
    recent = closes[-1]
    levels = sorted(set(closes), reverse=True)
    resistance = None
    support = None
    for lvl in levels:
        if lvl > recent and (resistance is None or lvl < resistance):
            resistance = lvl
        if lvl < recent and (support is None or lvl > support):
            support = lvl
    return recent, mid, support, resistance


def create_trend_tool(provider: DataProvider) -> Tool:
    async def analyze_trend(symbol: str, period: str = "3m") -> str:
        history = provider.get_history(symbol, period)
        if not history.data or len(history.data) < 20:
            return f"{symbol} 历史数据不足，需要至少 20 个交易日的数据。"

        closes = [d["close"] for d in history.data]
        volumes = [d["volume"] for d in history.data]
        name = history.name
        last_price = closes[-1]
        last_vol = volumes[-1]

        ma5 = _calc_ma(closes, 5)
        ma10 = _calc_ma(closes, 10)
        ma20 = _calc_ma(closes, 20)
        ma60 = _calc_ma(closes, 60) if len(closes) >= 60 else [None] * len(closes)
        dif, dea, macd_hist = _calc_macd(closes)
        rsi = _calc_rsi(closes, 14)
        vol_ma5 = _calc_ma(volumes, 5)

        cur_price, mid_price, support, resistance = _find_levels(closes)

        ma5_val = ma5[-1] or 0
        ma10_val = ma10[-1] or 0
        ma20_val = ma20[-1] or 0
        ma60_val = ma60[-1] if len(closes) >= 60 and ma60[-1] else 0
        rsi_val = rsi[-1] or 50
        macd_dif = dif[-1] or 0
        macd_dea = dea[-1] or 0
        macd_bar = macd_hist[-1] or 0
        vol_ratio = last_vol / (vol_ma5[-1] or 1)

        if ma5_val > ma10_val > ma20_val:
            trend = "多头排列，短期上涨趋势"
            base_scenario = "bullish"
        elif ma5_val < ma10_val < ma20_val:
            trend = "空头排列，短期下跌趋势"
            base_scenario = "bearish"
        else:
            trend = "均线交织，处于震荡整理"
            base_scenario = "neutral"

        overbought = rsi_val > 70
        oversold = rsi_val < 30
        vol_expanding = vol_ratio > 1.5
        vol_contracting = vol_ratio < 0.5

        bullish_cond = []
        neutral_cond = []
        bearish_cond = []

        if base_scenario == "bullish":
            bullish_cond.append("当前多头趋势延续")
            bullish_cond.append("成交量维持或放大")
            bullish_target = f"{last_price * 1.05:.2f}-{last_price * 1.10:.2f}"
            bearish_cond.append(f"跌破 20 日均线 ({ma20_val:.2f})")
            bearish_cond.append("MACD 死叉")
            bearish_target = f"{support or last_price * 0.95:.2f}"
            neutral_target = f"{last_price * 0.98:.2f}-{last_price * 1.02:.2f}"
        elif base_scenario == "bearish":
            bearish_cond.append("当前空头趋势延续")
            bearish_target = f"{support or last_price * 0.90:.2f}"
            bullish_cond.append(f"放量突破 10 日均线 ({ma10_val:.2f})")
            bullish_cond.append("MACD 金叉")
            bullish_target = f"{last_price * 1.03:.2f}-{last_price * 1.08:.2f}"
            neutral_target = f"{last_price * 0.98:.2f}-{last_price * 1.02:.2f}"
        else:
            bullish_cond.append("放量突破震荡区间上沿")
            r = resistance or last_price * 1.05
            bullish_target = f"{r:.2f}-{r * 1.05:.2f}"
            bearish_cond.append("放量跌破震荡区间下沿")
            s = support or last_price * 0.90
            bearish_target = f"{s:.2f}"
            neutral_cond.append("维持箱体震荡")
            neutral_target = f"{support or last_price * 0.95:.2f}-{resistance or last_price * 1.05:.2f}"

        if overbought:
            bearish_cond.append(f"RSI={rsi_val:.0f} 超买，回调风险")
        if oversold:
            bullish_cond.append(f"RSI={rsi_val:.0f} 超卖，反弹动能")
        if vol_expanding:
            bullish_cond.append("放量配合，趋势强化")
        if vol_contracting:
            neutral_cond.append("缩量整理，等待方向选择")

        sups = [f"{support:.2f}" if support else "N/A"]
        rests = [f"{resistance:.2f}" if resistance else "N/A"]

        return f"""📊 {name} ({symbol}) 技术面分析 — {period}周期

当前状态: {trend}
MACD: DIF={macd_dif:.2f} DEA={macd_dea:.2f} BAR={macd_bar:.2f}
RSI(14): {rsi_val:.1f}{' (超买)' if overbought else ' (超卖)' if oversold else ' (中性)'}
量比: {vol_ratio:.1f}{' (放量)' if vol_expanding else ' (缩量)' if vol_contracting else ' (正常)'}

┌─────────────────────────────────────────────────────┐
│ 🟢 乐观情景                                          │
│   条件: {'、'.join(bullish_cond) if bullish_cond else '基本面改善 + 市场情绪转好'}
│   目标: {bullish_target}
├─────────────────────────────────────────────────────┤
│ 🟡 中性情景                                          │
│   条件: {'、'.join(neutral_cond) if neutral_cond else '大盘横盘 + 无重大消息'}
│   区间: {neutral_target}
├─────────────────────────────────────────────────────┤
│ 🔴 悲观情景                                          │
│   条件: {'、'.join(bearish_cond) if bearish_cond else '大盘回落 + 板块利空'}
│   目标: {bearish_target}
└─────────────────────────────────────────────────────┘

支撑: {', '.join(sups)}    压力: {', '.join(rests)}

⚠️ 以上为基于技术指标的多情景推演，不构成投资建议。实际走势受政策、市场情绪、流动性等不可量化因素影响。"""

    return Tool(
        name="analyze_trend",
        description="对股票进行多情景技术分析。输出乐观/中性/悲观三种情景的走势推演。",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码"},
                "period": {"type": "string",
                           "description": "周期: 1m(1月) 3m(3月,默认) 6m(半年) 1y(1年)",
                           "default": "3m"},
            },
            "required": ["symbol"],
        },
        func=analyze_trend,
    )
