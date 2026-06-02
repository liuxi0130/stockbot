# 上证指数分析功能 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add index-level analysis capabilities to StockBot: market overview, technical trend analysis, and multi-horizon prediction for the Shanghai Composite Index (000001).

**Architecture:** New `stockbot/index/` module with three layers — `IndexDataProvider` (data abstraction for index quotes/breadth/sectors), `IndexAnalyzer` (pure-rule technical analysis producing multi-scenario narratives), `IndexPredictor` (rule-engine + optional LightGBM ML model → combined directional signals). Three new tools expose these to the LLM agent. ML model is optional; when absent, prediction degrades gracefully to rules-only mode.

**Tech Stack:** Python 3.11+, akshare, pandas, numpy, LightGBM (optional for ML), joblib

---

## File Map

```
stockbot/
├── index/
│   ├── __init__.py              # exports IndexDataProvider, IndexAnalyzer, IndexPredictor
│   ├── index_data.py            # IndexQuote, MarketBreadth, IndexDataProvider ABC, AkshareIndexProvider
│   ├── index_analyzer.py        # IndexAnalyzer — indicator calcs + multi-scenario narrative
│   └── index_predictor.py       # IndexPredictor — RuleEngine + MLPredictor + Combiner
├── tools/
│   ├── market_overview.py       # get_market_overview tool
│   ├── index_trend.py           # analyze_index tool
│   └── index_predict.py         # predict_index tool
scripts/
└── setup_index_model.py         # train LightGBM classifier for index direction prediction

tests/
├── test_index_data.py           # IndexDataProvider ABC + AkshareIndexProvider
├── test_index_analyzer.py       # IndexAnalyzer indicator calcs + scenario output
├── test_index_predictor.py      # RuleEngine + Combiner + MLPredictor
├── test_market_overview.py      # get_market_overview tool
├── test_index_trend.py          # analyze_index tool
└── test_index_predict.py        # predict_index tool

# Modified files:
stockbot/__init__.py             # register 3 new index tools in create_agent()
config.yaml                      # add index, index_model sections
```

---

## Phase 1: Index Data Layer

### Task 1: IndexDataProvider abstract + data classes

**Files:**
- Create: `stockbot/index/__init__.py`
- Create: `stockbot/index/index_data.py`
- Create: `tests/test_index_data.py`

- [ ] **Step 1: Write failing tests in tests/test_index_data.py**

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_index_data.py -v`
Expected: FAIL — Module not found

- [ ] **Step 3: Write stockbot/index/__init__.py**

```python
from stockbot.index.index_data import (
    IndexDataProvider, IndexQuote, MarketBreadth, AkshareIndexProvider,
)
from stockbot.index.index_analyzer import IndexAnalyzer
from stockbot.index.index_predictor import IndexPredictor

__all__ = [
    "IndexDataProvider", "IndexQuote", "MarketBreadth",
    "AkshareIndexProvider", "IndexAnalyzer", "IndexPredictor",
]
```

- [ ] **Step 4: Write stockbot/index/index_data.py**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class IndexQuote:
    code: str
    name: str
    price: float
    change_pct: float
    change_amt: float
    volume: float
    turnover: float          # 成交额（亿元）
    timestamp: str


@dataclass
class MarketBreadth:
    up_count: int             # 上涨家数
    down_count: int           # 下跌家数
    flat_count: int           # 平盘家数
    total_turnover: float     # 总成交额（亿元）
    limit_up: int             # 涨停家数
    limit_down: int           # 跌停家数


class IndexDataProvider(ABC):
    """Abstract interface for index/market data sources.
    
    Separate from stock DataProvider to avoid mixing index-level
    concepts (market breadth, sector rotation) with stock-level ones.
    """

    @abstractmethod
    def get_index_quote(self, index_code: str = "000001") -> IndexQuote:
        """Get real-time index quote. Default: 上证指数."""
        ...

    @abstractmethod
    def get_market_breadth(self) -> MarketBreadth:
        """Get market breadth: advance/decline counts, total turnover, limit up/down."""
        ...

    @abstractmethod
    def get_sector_performance(self, top_n: int = 5) -> list[dict]:
        """Get top/bottom sector performance. Returns list of {name, change_pct, leading_stock}."""
        ...

    @abstractmethod
    def get_index_history(self, index_code: str, period: str) -> list[dict]:
        """Get historical OHLCV data for the index. period: '1m','3m','6m','1y'."""
        ...

    @abstractmethod
    def get_index_news(self, limit: int = 5) -> list[dict]:
        """Get recent market/index news."""
        ...


class AkshareIndexProvider(IndexDataProvider):
    """A-share index data via akshare library."""

    def get_index_quote(self, index_code: str = "000001") -> IndexQuote:
        import akshare as ak
        from datetime import datetime
        try:
            df = ak.stock_zh_index_spot_em()
            row = df[df["代码"] == index_code]
            if row.empty:
                raise ValueError(f"未找到指数: {index_code}")
            r = row.iloc[0]
            return IndexQuote(
                code=index_code,
                name=r["名称"],
                price=float(r["最新价"]),
                change_pct=float(r["涨跌幅"]),
                change_amt=float(r["涨跌额"]),
                volume=float(r.get("成交量", 0)),
                turnover=float(r.get("成交额", 0)) / 1e8,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        except Exception as e:
            raise RuntimeError(f"获取指数行情失败: {e}")

    def get_market_breadth(self) -> MarketBreadth:
        import akshare as ak
        try:
            df = ak.stock_zh_index_spot_em()
            sh_row = df[df["代码"] == "000001"]
            if sh_row.empty:
                raise RuntimeError("上证指数数据不可用")
            r = sh_row.iloc[0]
            return MarketBreadth(
                up_count=int(r.get("上涨家数", 0)),
                down_count=int(r.get("下跌家数", 0)),
                flat_count=int(r.get("平盘家数", 0)),
                total_turnover=float(r.get("成交额", 0)) / 1e8,
                limit_up=0,
                limit_down=0,
            )
        except Exception as e:
            raise RuntimeError(f"获取市场宽度失败: {e}")

    def get_sector_performance(self, top_n: int = 5) -> list[dict]:
        import akshare as ak
        try:
            df = ak.stock_board_industry_name_em()
            df_sorted = df.sort_values("涨跌幅", ascending=False)
            top = df_sorted.head(top_n)
            bottom = df_sorted.tail(top_n)
            result = []
            for _, row in top.iterrows():
                result.append({
                    "name": row["板块名称"],
                    "change_pct": float(row["涨跌幅"]),
                    "leading_stock": row.get("领涨股票", ""),
                    "rank": "top",
                })
            for _, row in bottom.iterrows()[::-1]:
                result.append({
                    "name": row["板块名称"],
                    "change_pct": float(row["涨跌幅"]),
                    "leading_stock": row.get("领涨股票", ""),
                    "rank": "bottom",
                })
            return result
        except Exception:
            return []

    def get_index_history(self, index_code: str, period: str = "3m") -> list[dict]:
        import akshare as ak
        period_days = {"1m": 30, "3m": 90, "6m": 180, "1y": 250}
        days = period_days.get(period, 90)
        try:
            df = ak.stock_zh_index_daily_em(symbol=f"sh{index_code}")
            name_map = {
                "date": "date", "open": "open", "high": "high",
                "low": "low", "close": "close", "volume": "volume",
            }
            data = []
            for _, row in df.tail(days).iterrows():
                data.append({
                    "date": str(row["date"])[:10],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                })
            return data
        except Exception as e:
            raise RuntimeError(f"获取指数历史数据失败: {e}")

    def get_index_news(self, limit: int = 5) -> list[dict]:
        import akshare as ak
        try:
            df = ak.stock_news_em(symbol="000001")
            if df.empty:
                return []
            col_map = {}
            for c in df.columns:
                if "标题" in c:
                    col_map["title"] = c
                elif "来源" in c:
                    col_map["source"] = c
                elif "时间" in c:
                    col_map["time"] = c
                elif "链接" in c:
                    col_map["url"] = c
            return [
                {
                    "title": row.get(col_map.get("title", ""), ""),
                    "source": row.get(col_map.get("source", ""), ""),
                    "time": str(row.get(col_map.get("time", ""), "")),
                    "url": row.get(col_map.get("url", ""), ""),
                }
                for _, row in df.head(limit).iterrows()
            ]
        except Exception:
            return []
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_index_data.py -v`
Expected: 5 PASS (MockIndexProvider tests pass against the abstract interface)

- [ ] **Step 6: Commit**

```bash
git add stockbot/index/__init__.py stockbot/index/index_data.py tests/test_index_data.py
git commit -m "feat: add IndexDataProvider abstract and AkshareIndexProvider"
```

---

## Phase 2: Index Analyzer (Technical Analysis)

### Task 2: IndexAnalyzer — multi-scenario technical analysis for indices

**Files:**
- Create: `stockbot/index/index_analyzer.py`
- Create: `tests/test_index_analyzer.py`

- [ ] **Step 1: Write failing tests in tests/test_index_analyzer.py**

```python
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
        assert result.ma5 > result.ma10  # short-term MA should be above

    def test_downward_trend_detected(self):
        analyzer = IndexAnalyzer()
        data = make_sample_ohlcv(60, "down")
        result = analyzer.analyze(data, index_name="上证指数", index_code="000001")
        assert result.base_scenario in ("bearish", "neutral")
        assert result.rsi < 60

    def test_insufficient_data_raises(self):
        analyzer = IndexAnalyzer()
        data = make_sample_ohlcv(10, "up")
        with pytest.raises(ValueError, match="至少需要"):
            analyzer.analyze(data, index_name="上证指数", index_code="000001")

    def test_result_contains_support_resistance(self):
        analyzer = IndexAnalyzer()
        data = make_sample_ohlcv(60, "up")
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
        assert "乐观" in output or "中性" in output or "悲观" in output
        assert "⚠️" in output or "仅供参考" in output or "投资建议" in output
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_index_analyzer.py -v`
Expected: FAIL — Module not found

- [ ] **Step 3: Write stockbot/index/index_analyzer.py**

```python
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
        vol_ratio = last_close / (vol_ma5[-1] or 1) if last_close else 1.0

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
            neutral_cond = ["维持箱体震荡"]
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_index_analyzer.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add stockbot/index/index_analyzer.py tests/test_index_analyzer.py
git commit -m "feat: add IndexAnalyzer with multi-scenario technical analysis"
```

---

## Phase 3: Index Predictor (Rules + ML + Combiner)

### Task 3: IndexPredictor — RuleEngine + MLPredictor + Combiner

**Files:**
- Create: `stockbot/index/index_predictor.py`
- Create: `tests/test_index_predictor.py`

- [ ] **Step 1: Write failing tests in tests/test_index_predictor.py**

```python
import pytest
from stockbot.index.index_predictor import (
    IndexPredictor, RuleEngine, IndexPrediction, IndexSignal,
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
        rule_engine = RuleEngine()
        rule_pred = rule_engine.evaluate(data)
        # When ML is absent, combiner uses rule-only
        from stockbot.index.index_predictor import _combine
        combined = _combine(rule_pred, ml_pred=None)
        assert combined.short_term_score == rule_pred.short_term_score

    def test_signal_to_str(self):
        from stockbot.index.index_predictor import IndexSignal
        s = IndexPredictor._signal_to_str(IndexSignal.STRONG_BUY)
        assert "看多" in s
        assert "🟢" in s

    def test_score_to_signal(self):
        from stockbot.index.index_predictor import IndexSignal
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
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_index_predictor.py -v`
Expected: FAIL — Module not found

- [ ] **Step 3: Write stockbot/index/index_predictor.py**

```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
    """Raw prediction from a single engine (rule or ML)."""
    short_term_score: int      # 0-100
    mid_term_score: int        # 0-100
    short_term_signal: IndexSignal
    mid_term_signal: IndexSignal
    short_term_detail: str     # human-readable rationale
    mid_term_detail: str
    source: str                # "rule" | "ml"


class RuleEngine:
    """Deterministic technical-rule-based prediction engine."""

    def __init__(self):
        self.analyzer = IndexAnalyzer()

    def evaluate(self, ohlcv: list[dict]) -> IndexPrediction:
        """Compute directional prediction from technical indicators."""
        analysis = self.analyzer.analyze(ohlcv)

        # Score components (each 0-25, total 0-100)
        score = 50  # start neutral

        # Trend (0-25 pts)
        if analysis.base_scenario == "bullish":
            score += 20
        elif analysis.base_scenario == "bearish":
            score -= 20
        # neutral → no change

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

        # Mid-term: same logic but weighted toward trend/MACD
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
    """Optional LightGBM-based prediction model for index direction.

    Model format: a joblib file containing a dict with keys:
    - 'short_model': sklearn-compatible classifier
    - 'mid_model': sklearn-compatible classifier
    - 'feature_names': list[str]
    """

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
        """Extract technical features from OHLCV data."""
        closes = [d["close"] for d in ohlcv]
        volumes = [d["volume"] for d in ohlcv]
        highs = [d["high"] for d in ohlcv]
        lows = [d["low"] for d in ohlcv]
        n = len(closes)

        # Return at 5/10/20 days
        ret5 = (closes[-1] / closes[-6] - 1) if n >= 6 else 0
        ret10 = (closes[-1] / closes[-11] - 1) if n >= 11 else 0
        ret20 = (closes[-1] / closes[-21] - 1) if n >= 21 else 0

        # Volatility
        import statistics
        vol20 = statistics.stdev(
            [(closes[i] / closes[i-1] - 1) for i in range(max(1, n-20), n)]
        ) if n >= 21 else 0

        # Volume trend
        if n >= 11:
            vol_ma5 = sum(volumes[-5:]) / 5
            vol_ma10 = sum(volumes[-10:]) / 10
            vol_ratio = vol_ma5 / vol_ma10 if vol_ma10 > 0 else 1.0
        else:
            vol_ratio = 1.0

        # MA positions
        ma5 = sum(closes[-5:]) / 5 if n >= 5 else closes[-1]
        ma10 = sum(closes[-10:]) / 10 if n >= 10 else closes[-1]
        ma20 = sum(closes[-20:]) / 20 if n >= 20 else closes[-1]
        price_vs_ma20 = (closes[-1] / ma20 - 1) if ma20 > 0 else 0

        # RSI-like simple ratio
        if n >= 15:
            gains = sum(max(closes[i] - closes[i-1], 0) for i in range(n-14, n))
            losses = sum(max(closes[i-1] - closes[i], 0) for i in range(n-14, n))
            rsi_simple = 100 * gains / (gains + losses) if (gains + losses) > 0 else 50
        else:
            rsi_simple = 50

        # High-low range
        range_ratio = (
            (highs[-1] - lows[-1]) / closes[-1] if closes[-1] > 0 else 0
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
    """Multi-horizon index direction predictor.

    Combines rule-based signals with optional ML model predictions.
    Degrades gracefully to rules-only when ML model is not available.
    """

    def __init__(self, index_provider, model_dir: str = "data/index_model",
                 ml_enabled: bool = False):
        self.index_provider = index_provider
        self.rule_engine = RuleEngine()
        self.ml_enabled = ml_enabled
        self.ml_predictor = MLPredictor(model_dir) if ml_enabled else None

    def is_ml_available(self) -> bool:
        return self.ml_predictor is not None and self.ml_predictor.is_available()

    def predict(self, index_code: str = "000001", period: str = "3m") -> str:
        """Run prediction and return formatted output string."""
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

    @staticmethod
    def _score_to_signal(score: int) -> IndexSignal:
        return _score_to_signal(score)  # delegate to module-level function

    @staticmethod
    def _signal_to_str(signal: IndexSignal) -> str:
        return {
            IndexSignal.STRONG_BUY: "🟢 强烈看多",
            IndexSignal.MILD_BUY: "🟡 温和看多",
            IndexSignal.NEUTRAL: "⚪ 中性/震荡",
            IndexSignal.MILD_SELL: "🟠 温和看空",
            IndexSignal.STRONG_SELL: "🔴 强烈看空",
        }[signal]


# ── module-level helpers ──────────────────────────────────

def _score_to_signal(score: int) -> IndexSignal:
    """Standalone signal classifier used by RuleEngine and MLPredictor."""
    if score >= 80:
        return IndexSignal.STRONG_BUY
    elif score >= 60:
        return IndexSignal.MILD_BUY
    elif score >= 40:
        return IndexSignal.NEUTRAL
    elif score >= 20:
        return IndexSignal.MILD_SELL
    return IndexSignal.STRONG_SELL


def _combine(rule: IndexPrediction, ml: Optional[IndexPrediction] = None) -> IndexPrediction:
    """Combine rule and ML predictions into a single output."""
    if ml is None:
        return rule  # rules only

    short_score = int(rule.short_term_score * 0.5 + ml.short_term_score * 0.5)
    mid_score = int(rule.mid_term_score * 0.5 + ml.mid_term_score * 0.5)

    return IndexPrediction(
        short_term_score=short_score,
        mid_term_score=mid_score,
        short_term_signal=_score_to_signal(short_score),
        mid_term_signal=_score_to_signal(mid_score),
        short_term_detail=(
            f"规则({rule.short_term_score}) + ML({ml.short_term_score})"
        ),
        mid_term_detail=(
            f"规则({rule.mid_term_score}) + ML({ml.mid_term_score})"
        ),
        source="combined",
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_index_predictor.py -v`
Expected: 9 PASS

- [ ] **Step 5: Commit**

```bash
git add stockbot/index/index_predictor.py tests/test_index_predictor.py
git commit -m "feat: add IndexPredictor with RuleEngine + MLPredictor + Combiner"
```

---

## Phase 4: Three Index Tools

### Task 4: get_market_overview tool

**Files:**
- Create: `stockbot/tools/market_overview.py`
- Create: `tests/test_market_overview.py`

- [ ] **Step 1: Write failing tests in tests/test_market_overview.py**

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_market_overview.py -v`
Expected: FAIL — Module not found

- [ ] **Step 3: Write stockbot/tools/market_overview.py**

```python
from stockbot.tools.base import Tool
from stockbot.index.index_data import IndexDataProvider


def create_market_overview_tool(provider: IndexDataProvider) -> Tool:
    async def get_market_overview() -> str:
        try:
            quote = provider.get_index_quote("000001")
        except Exception as e:
            return f"获取上证指数行情失败: {e}"

        try:
            breadth = provider.get_market_breadth()
        except Exception:
            breadth = None

        try:
            sectors = provider.get_sector_performance(top_n=5)
        except Exception:
            sectors = []

        arrow = "↑" if quote.change_pct >= 0 else "↓"
        sign = "+" if quote.change_pct >= 0 else ""

        lines = [
            f"📈 大盘概况 — {quote.timestamp}",
            "",
            f"  {quote.name} ({quote.code})",
            f"  最新点位: {quote.price:.2f}",
            f"  涨跌幅: {sign}{quote.change_pct:.2f}% {arrow}",
            f"  涨跌额: {sign}{quote.change_amt:.2f}",
            f"  成交额: {quote.turnover:.0f} 亿元",
        ]

        if breadth:
            total = breadth.up_count + breadth.down_count + breadth.flat_count
            up_pct = breadth.up_count / total * 100 if total > 0 else 0
            down_pct = breadth.down_count / total * 100 if total > 0 else 0
            lines.extend([
                "",
                f"📊 市场宽度",
                f"  上涨: {breadth.up_count} 家 ({up_pct:.1f}%)",
                f"  下跌: {breadth.down_count} 家 ({down_pct:.1f}%)",
                f"  平盘: {breadth.flat_count} 家",
                f"  两市成交额: {breadth.total_turnover:.0f} 亿元",
                f"  涨停: {breadth.limit_up} 家  |  跌停: {breadth.limit_down} 家",
            ])

        if sectors:
            lines.append("")
            lines.append("🔥 行业板块")
            top_sectors = [s for s in sectors if s.get("rank") == "top"]
            bottom_sectors = [s for s in sectors if s.get("rank") == "bottom"]
            lines.append("  涨幅居前:")
            for s in top_sectors:
                lines.append(
                    f"    🟢 {s['name']}: +{s['change_pct']:.2f}%"
                    + (f"  (领涨: {s.get('leading_stock', '')})" if s.get("leading_stock") else "")
                )
            lines.append("  跌幅居前:")
            for s in bottom_sectors:
                lines.append(
                    f"    🔴 {s['name']}: {s['change_pct']:.2f}%"
                    + (f"  (领跌: {s.get('leading_stock', '')})" if s.get("leading_stock") else "")
                )

        lines.extend([
            "",
            "⚠️ 以上仅为市场概况数据展示，不构成投资建议。",
        ])

        return "\n".join(lines)

    return Tool(
        name="get_market_overview",
        description=(
            "获取A股大盘概况。返回上证指数点位、涨跌幅、市场宽度(涨跌家数)、"
            "两市成交额、行业板块热度等信息。无需参数，适合在市场分析或个股"
            "分析前先了解整体市场环境。"
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        func=get_market_overview,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_market_overview.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add stockbot/tools/market_overview.py tests/test_market_overview.py
git commit -m "feat: add get_market_overview tool for index breadth and sector heat"
```

---

### Task 5: analyze_index tool

**Files:**
- Create: `stockbot/tools/index_trend.py`
- Create: `tests/test_index_trend.py`

- [ ] **Step 1: Write failing tests in tests/test_index_trend.py**

```python
import pytest
from stockbot.index.index_data import IndexQuote, MarketBreadth
from stockbot.tools.index_trend import create_index_trend_tool


class MockIndexProvider:
    def get_index_quote(self, index_code: str = "000001") -> IndexQuote:
        return IndexQuote(
            code=index_code, name="上证指数", price=3350.5,
            change_pct=0.85, change_amt=28.2, volume=1.5e9,
            turnover=3200.0, timestamp="2026-06-02",
        )

    def get_index_history(self, index_code: str, period: str) -> list[dict]:
        import random
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
        assert "乐观" in result or "中性" in result or "悲观" in result

    @pytest.mark.asyncio
    async def test_trend_contains_warning(self):
        provider = MockIndexProvider()
        tool = create_index_trend_tool(provider)
        result = await tool.run(index_code="000001", period="1m")
        assert "⚠️" in result or "仅供参考" in result or "投资建议" in result

    @pytest.mark.asyncio
    async def test_trend_insufficient_data(self):
        provider = MockIndexProvider()
        # Override to return very little data
        provider.get_index_history = lambda *a, **kw: [
            {"date": f"2026-05-{20+i:02d}", "open": 3300, "high": 3320,
             "low": 3290, "close": 3310, "volume": 1e9}
            for i in range(5)
        ]
        tool = create_index_trend_tool(provider)
        result = await tool.run(index_code="000001", period="1m")
        assert "不足" in result or "至少" in result
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_index_trend.py -v`
Expected: FAIL — Module not found

- [ ] **Step 3: Write stockbot/tools/index_trend.py**

```python
from stockbot.tools.base import Tool
from stockbot.index.index_data import IndexDataProvider
from stockbot.index.index_analyzer import IndexAnalyzer


INDEX_NAMES = {
    "000001": "上证指数",
    "399001": "深证成指",
    "399006": "创业板指",
}


def create_index_trend_tool(provider: IndexDataProvider) -> Tool:
    analyzer = IndexAnalyzer()

    async def analyze_index(index_code: str = "000001", period: str = "3m") -> str:
        index_name = INDEX_NAMES.get(index_code, f"指数{index_code}")

        try:
            ohlcv = provider.get_index_history(index_code, period)
        except Exception as e:
            return f"获取{index_name}历史数据失败: {e}"

        try:
            result = analyzer.analyze(ohlcv, index_name=index_name, index_code=index_code)
        except ValueError as e:
            return str(e)

        return analyzer.format_output(result, period)

    return Tool(
        name="analyze_index",
        description=(
            "对大盘指数进行多情景技术分析。输入指数代码(默认000001上证指数)"
            "和周期(1m/3m/6m/1y)，输出乐观/中性/悲观三种情景推演，"
            "包含MA均线、MACD、RSI、量比等指标。支持上证指数、深证成指、"
            "创业板指等主要指数。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "index_code": {
                    "type": "string",
                    "description": "指数代码: 000001(上证指数), 399001(深证成指), 399006(创业板指)",
                    "default": "000001",
                },
                "period": {
                    "type": "string",
                    "description": "分析周期: 1m(1月), 3m(3月,默认), 6m(半年), 1y(1年)",
                    "default": "3m",
                },
            },
            "required": [],
        },
        func=analyze_index,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_index_trend.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add stockbot/tools/index_trend.py tests/test_index_trend.py
git commit -m "feat: add analyze_index tool for index technical analysis"
```

---

### Task 6: predict_index tool

**Files:**
- Create: `stockbot/tools/index_predict.py`
- Create: `tests/test_index_predict.py`

- [ ] **Step 1: Write failing tests in tests/test_index_predict.py**

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_index_predict.py -v`
Expected: FAIL — Module not found

- [ ] **Step 3: Write stockbot/tools/index_predict.py**

```python
from stockbot.tools.base import Tool


def create_index_predict_tool(predictor) -> Tool:
    """Create a predict_index tool backed by *predictor* (IndexPredictor instance)."""

    async def predict_index(index_code: str = "000001", period: str = "3m") -> str:
        return predictor.predict(index_code, period)

    return Tool(
        name="predict_index",
        description=(
            "预测大盘指数短期和中期走势方向。基于技术规则和可选机器学习模型，"
            "输出综合评分(0-100)和方向信号(看多/中性/看空)。"
            "支持上证指数(000001)、深证成指(399001)、创业板指(399006)。"
            "注意: ML增强模型可能需要额外配置，未配置时使用纯规则预测。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "index_code": {
                    "type": "string",
                    "description": "指数代码: 000001(上证指数), 399001(深证成指), 399006(创业板指)",
                    "default": "000001",
                },
                "period": {
                    "type": "string",
                    "description": "数据周期: 1m(1月), 3m(3月,默认), 6m(半年), 1y(1年)",
                    "default": "3m",
                },
            },
            "required": [],
        },
        func=predict_index,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_index_predict.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add stockbot/tools/index_predict.py tests/test_index_predict.py
git commit -m "feat: add predict_index tool for multi-horizon index prediction"
```

---

## Phase 5: Integration

### Task 7: Wire index tools into create_agent()

**Files:**
- Modify: `stockbot/__init__.py`
- Modify: `config.yaml`

- [ ] **Step 1: Update config.yaml**

Add the following sections to `config.yaml` after the `qlib` section:

```yaml
index:
  default_code: "000001"
  supported_codes:
    - "000001"
    - "399001"
    - "399006"

index_model:
  model_dir: data/index_model
  short_term_horizon: 3
  mid_term_horizon: 20
```

- [ ] **Step 2: Modify stockbot/__init__.py**

Add the following import block after line 12 (`from stockbot.quant.predictor import QuantPredictor`):

```python
from stockbot.index.index_data import AkshareIndexProvider
from stockbot.index.index_predictor import IndexPredictor
from stockbot.tools.market_overview import create_market_overview_tool
from stockbot.tools.index_trend import create_index_trend_tool
from stockbot.tools.index_predict import create_index_predict_tool
```

Then add after the qlib config block (line ~73, after the qlib logger.info), insert:

```python
    # ── Index analysis tools ──────────────────────────────
    index_provider = AkshareIndexProvider()
    tool_registry.register(create_market_overview_tool(index_provider))
    tool_registry.register(create_index_trend_tool(index_provider))

    index_model_cfg = cfg.get("index_model", {})
    index_model_dir = index_model_cfg.get("model_dir", "data/index_model")
    ml_enabled = Path(index_model_dir).exists()
    index_predictor = IndexPredictor(
        index_provider=index_provider,
        model_dir=index_model_dir,
        ml_enabled=ml_enabled,
    )
    tool_registry.register(create_index_predict_tool(index_predictor))
    if ml_enabled:
        LOGGER.info("Index ML model loaded from %s", index_model_dir)
    else:
        LOGGER.info("Index ML model not found; using rules-only prediction")
```

- [ ] **Step 3: Verify integration — run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: All existing 70 tests + new tests = ~97 tests, all PASS

- [ ] **Step 4: Commit**

```bash
git add stockbot/__init__.py config.yaml
git commit -m "feat: integrate 3 index tools (overview, trend, predict) into agent"
```

---

## Phase 6: ML Model Training Script

### Task 8: setup_index_model.py — train LightGBM for index prediction

**Files:**
- Create: `scripts/setup_index_model.py`

- [ ] **Step 1: Write scripts/setup_index_model.py**

```python
#!/usr/bin/env python
"""Train LightGBM classifiers for index direction prediction.

Usage:
    python scripts/setup_index_model.py               # full setup
    python scripts/setup_index_model.py --index 000001  # specify index

Trains two classifiers:
- Short-term: predict whether index goes up in next 3 days
- Mid-term:   predict whether index goes up in next 20 days

Saves the trained model to data/index_model/index_model.pkl.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "index_model"


def fetch_index_data(index_code: str = "000001") -> pd.DataFrame:
    """Fetch historical index OHLCV data via akshare."""
    import akshare as ak

    LOGGER.info("Fetching index %s data ...", index_code)
    df = ak.stock_zh_index_daily_em(symbol=f"sh{index_code}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def build_features(df: pd.DataFrame, short_horizon: int = 3,
                   mid_horizon: int = 20) -> pd.DataFrame:
    """Construct features and labels for both horizons."""
    df = df.copy()

    closes = df["close"].values
    volumes = df["volume"].values
    highs = df["high"].values
    lows = df["low"].values

    n = len(df)
    features = []

    for i in range(n):
        if i < 30:
            continue

        slice_c = closes[:i + 1]
        slice_v = volumes[:i + 1]
        k = len(slice_c)

        ret5 = (slice_c[-1] / slice_c[-6] - 1) if k >= 6 else 0
        ret10 = (slice_c[-1] / slice_c[-11] - 1) if k >= 11 else 0
        ret20 = (slice_c[-1] / slice_c[-21] - 1) if k >= 21 else 0

        # 20-day volatility
        rets = [(slice_c[t] / slice_c[t-1] - 1) for t in range(max(1, k-20), k)]
        vol20 = float(np.std(rets)) if len(rets) > 1 else 0

        # Volume ratio
        vol_ma5 = float(np.mean(slice_v[-5:])) if k >= 5 else slice_v[-1]
        vol_ma10 = float(np.mean(slice_v[-10:])) if k >= 10 else slice_v[-1]
        vol_ratio = vol_ma5 / vol_ma10 if vol_ma10 > 0 else 1.0

        # MA positions
        ma5 = float(np.mean(slice_c[-5:])) if k >= 5 else slice_c[-1]
        ma10 = float(np.mean(slice_c[-10:])) if k >= 10 else slice_c[-1]
        ma20 = float(np.mean(slice_c[-20:])) if k >= 20 else slice_c[-1]
        price_vs_ma20 = slice_c[-1] / ma20 - 1 if ma20 > 0 else 0
        ma5_vs_ma20 = ma5 / ma20 - 1 if ma20 > 0 else 0

        # Simple RSI
        if k >= 15:
            gains = sum(
                max(slice_c[t] - slice_c[t-1], 0) for t in range(k-14, k)
            )
            losses = sum(
                max(slice_c[t-1] - slice_c[t], 0) for t in range(k-14, k)
            )
            rsi = 100 * gains / (gains + losses) if (gains + losses) > 0 else 50
        else:
            rsi = 50

        # Range
        high_low_range = (highs[i] - lows[i]) / slice_c[-1] if slice_c[-1] > 0 else 0

        # Labels
        short_up = 0
        mid_up = 0
        if i + short_horizon < n:
            short_up = 1 if closes[i + short_horizon] > closes[i] else 0
        if i + mid_horizon < n:
            mid_up = 1 if closes[i + mid_horizon] > closes[i] else 0

        features.append([
            ret5, ret10, ret20, vol20, vol_ratio,
            price_vs_ma20, ma5_vs_ma20, rsi / 100,
            high_low_range, short_up, mid_up,
        ])

    cols = [
        "ret5", "ret10", "ret20", "vol20", "vol_ratio",
        "price_vs_ma20", "ma5_vs_ma20", "rsi_norm",
        "high_low_range", "short_label", "mid_label",
    ]
    return pd.DataFrame(features, columns=cols)


def train_models(df: pd.DataFrame) -> dict:
    """Train short-term and mid-term LGBM classifiers."""
    feature_cols = [
        "ret5", "ret10", "ret20", "vol20", "vol_ratio",
        "price_vs_ma20", "ma5_vs_ma20", "rsi_norm", "high_low_range",
    ]

    X = df[feature_cols].values
    y_short = df["short_label"].values
    y_mid = df["mid_label"].values

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_short, test_size=0.2, shuffle=False,
    )
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X, y_mid, test_size=0.2, shuffle=False,
    )

    LOGGER.info("Training short-term model (horizon=3 days) ...")
    short_model = LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=42, verbose=-1,
    )
    short_model.fit(X_train_s, y_train_s)
    short_acc = short_model.score(X_test_s, y_test_s)
    LOGGER.info("  Short-term accuracy: %.2f%%", short_acc * 100)

    LOGGER.info("Training mid-term model (horizon=20 days) ...")
    mid_model = LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=42, verbose=-1,
    )
    mid_model.fit(X_train_m, y_train_m)
    mid_acc = mid_model.score(X_test_m, y_test_m)
    LOGGER.info("  Mid-term accuracy: %.2f%%", mid_acc * 100)

    return {
        "short_model": short_model,
        "mid_model": mid_model,
        "feature_names": feature_cols,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train index prediction models for StockBot",
    )
    parser.add_argument("--index", default="000001",
                        help="Index code (default: 000001 = 上证指数)")
    parser.add_argument("--short-horizon", type=int, default=3)
    parser.add_argument("--mid-horizon", type=int, default=20)
    args = parser.parse_args()

    df = fetch_index_data(args.index)
    LOGGER.info("Fetched %d rows for index %s", len(df), args.index)

    features = build_features(df, args.short_horizon, args.mid_horizon)
    LOGGER.info("Built %d feature vectors", len(features))

    models = train_models(features)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "index_model.pkl"
    joblib.dump(models, model_path)
    LOGGER.info(
        "Model saved → %s (%.1f MB)",
        model_path, model_path.stat().st_size / 1024 / 1024,
    )
    LOGGER.info(
        "Setup complete! Index prediction ML model is ready. "
        "Restart StockBot to use it."
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/setup_index_model.py
git commit -m "feat: add setup_index_model.py for training index prediction models"
```

---

## Phase 7: Final Verification

### Task 9: Run all tests and verify

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS (~97 tests)

- [ ] **Step 2: Verify import chain**

Run:
```
python -c "from stockbot.index.index_data import AkshareIndexProvider; print('index_data OK')"
python -c "from stockbot.index.index_analyzer import IndexAnalyzer; print('index_analyzer OK')"
python -c "from stockbot.index.index_predictor import IndexPredictor; print('index_predictor OK')"
python -c "from stockbot.tools.market_overview import create_market_overview_tool; print('market_overview OK')"
python -c "from stockbot.tools.index_trend import create_index_trend_tool; print('index_trend OK')"
python -c "from stockbot.tools.index_predict import create_index_predict_tool; print('index_predict OK')"
```

Expected: 6 "OK" lines

- [ ] **Step 3: Verify create_agent() still works**

Run: `python -c "from stockbot import create_agent; a,s,_,c = create_agent(); print('create_agent OK, tools:', len(a.tool_registry._tools))"`
Expected: "create_agent OK, tools: 9" (6 stock + 3 index)

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: finalize index analysis feature integration"
```
