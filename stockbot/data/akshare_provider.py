import akshare as ak
from datetime import datetime
from stockbot.data.base import DataProvider, StockQuote, StockHistory


class AkshareProvider(DataProvider):
    """A-share stock data via akshare library."""

    def search(self, query: str) -> list[dict]:
        try:
            df = ak.stock_info_a_code_name()
            mask = df["name"].str.contains(query) | df["code"].str.contains(query)
            results = df[mask].head(10)
            return [
                {"symbol": row["code"], "name": row["name"],
                 "market": "SH" if row["code"].startswith("6") else "SZ"}
                for _, row in results.iterrows()
            ]
        except Exception:
            return []

    def _to_xq_symbol(self, symbol: str) -> str:
        """Convert 600519 → SH600519 for Xueqiu API."""
        if symbol.startswith(("6", "68", "9")):
            return f"SH{symbol}"
        return f"SZ{symbol}"

    def _to_sina_symbol(self, symbol: str) -> str:
        """Convert 600519 → sh600519 for Sina API."""
        if symbol.startswith(("6", "68", "9")):
            return f"sh{symbol}"
        return f"sz{symbol}"

    def get_realtime(self, symbol: str) -> StockQuote:
        try:
            df = ak.stock_individual_spot_xq(symbol=self._to_xq_symbol(symbol))
            items = {row["item"]: row["value"] for _, row in df.iterrows()}
            return StockQuote(
                symbol=symbol,
                name=items.get("名称", symbol),
                price=float(items.get("现价", 0)),
                change_pct=float(items.get("涨跌幅", 0)),
                volume=float(items.get("成交量", 0)),
                timestamp=str(items.get("时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
            )
        except Exception as e:
            raise RuntimeError(f"获取行情失败: {e}")

    def get_history(self, symbol: str, period: str = "3m") -> StockHistory:
        period_days = {"1m": 30, "3m": 90, "6m": 180, "1y": 250}
        days = period_days.get(period, 90)
        try:
            df = ak.stock_zh_a_daily(symbol=self._to_sina_symbol(symbol), adjust="qfq")
            name = self._get_name(symbol)
            data = [
                {
                    "date": str(row["date"])[:10],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
                for _, row in df.tail(days).iterrows()
            ]
            return StockHistory(symbol=symbol, name=name, data=data)
        except Exception as e:
            raise RuntimeError(f"获取历史数据失败: {e}")

    def get_financial(self, symbol: str) -> dict:
        try:
            df = ak.stock_financial_abstract_ths(symbol=symbol)
            if df.empty:
                return {}
            latest = df.iloc[-1]
            return {
                "pe": None,
                "pb": None,
                "roe": self._safe_float(latest.get("净资产收益率")),
                "revenue_growth": self._safe_float(latest.get("营业总收入同比增长率")),
                "eps": self._safe_float(latest.get("基本每股收益")),
                "net_profit": self._safe_float(latest.get("净利润")),
                "net_profit_growth": self._safe_float(latest.get("净利润同比增长率")),
            }
        except Exception as e:
            raise RuntimeError(f"获取财务数据失败: {e}")

    def get_news(self, symbol: str, limit: int = 5) -> list[dict]:
        try:
            import pandas as pd
            # stock_news_em uses 　 regex which is incompatible with
            # ArrowStringArray in pandas 3.x. Disable Arrow strings for this call.
            with pd.option_context("future.infer_string", False):
                df = ak.stock_news_em(symbol=symbol)
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
                elif "链接" in c or "url" in c:
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

    def _get_name(self, symbol: str) -> str:
        results = self.search(symbol)
        return results[0]["name"] if results else symbol

    @staticmethod
    def _safe_float(value) -> float | None:
        if value is None or value == "-" or value == "" or isinstance(value, bool):
            return None
        try:
            s = str(value)
            multiplier = 1.0
            if s.endswith("%"):
                s = s[:-1]
            elif "亿" in s:
                s = s.replace("亿", "")
                multiplier = 100_000_000
            elif "万" in s:
                s = s.replace("万", "")
                multiplier = 10_000
            return float(s) * multiplier
        except (ValueError, TypeError):
            return None
