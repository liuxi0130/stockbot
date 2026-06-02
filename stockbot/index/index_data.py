"""Index/market data abstraction and Akshare implementation.

ABC and implementation are co-located in this single module for simplicity
(single-provider case). If additional providers are added, the ABC should be
split into its own module under stockbot/index/abstract.py.
"""

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
                limit_up=0,      # Not populated by AkshareIndexProvider (spot endpoint
	                             # from akshare does not provide limit up/down data)
                limit_down=0,    # See note above
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
            for _, row in list(bottom.iterrows())[::-1]:
                result.append({
                    "name": row["板块名称"],
                    "change_pct": float(row["涨跌幅"]),
                    "leading_stock": row.get("领涨股票", ""),
                    "rank": "bottom",
                })
            return result
        except Exception as e:
            raise RuntimeError(f"获取板块表现数据失败: {e}")

    def get_index_history(self, index_code: str, period: str = "3m") -> list[dict]:
        import akshare as ak
        period_days = {"1m": 30, "3m": 90, "6m": 180, "1y": 250}
        days = period_days.get(period, 90)
        prefix = "sz" if not index_code.startswith(("6", "68", "9")) else "sh"
        try:
            df = ak.stock_zh_index_daily_em(symbol=f"{prefix}{index_code}")
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
            # Column mapping uses fuzzy name matching because akshare's column
            # names are in Chinese and may vary across versions. We search for
            # substrings like "标题", "来源", "时间", "链接" to stay version-resilient.
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
            raise RuntimeError("获取指数新闻失败")
