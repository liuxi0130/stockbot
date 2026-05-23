import akshare as ak
from datetime import datetime
from stockbot.data.base import DataProvider, StockQuote, StockHistory


class AkshareProvider(DataProvider):
    """A-share stock data via akshare library."""

    def search(self, query: str) -> list[dict]:
        try:
            df = ak.stock_info_a_code_name()
            mask = df["名称"].str.contains(query) | df["代码"].str.contains(query)
            results = df[mask].head(10)
            return [
                {"symbol": row["代码"], "name": row["名称"],
                 "market": "SH" if row["代码"].startswith("6") else "SZ"}
                for _, row in results.iterrows()
            ]
        except Exception:
            return []

    def get_realtime(self, symbol: str) -> StockQuote:
        try:
            df = ak.stock_zh_a_spot_em()
            row = df[df["代码"] == symbol]
            if row.empty:
                raise ValueError(f"未找到股票: {symbol}")
            r = row.iloc[0]
            return StockQuote(
                symbol=symbol,
                name=r["名称"],
                price=float(r["最新价"]),
                change_pct=float(r["涨跌幅"]),
                volume=float(r["成交量"]),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        except Exception as e:
            raise RuntimeError(f"获取行情失败: {e}")

    def get_history(self, symbol: str, period: str = "3m") -> StockHistory:
        period_days = {"1m": 30, "3m": 90, "6m": 180, "1y": 250}
        days = period_days.get(period, 90)
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
            name = self._get_name(symbol)
            data = [
                {
                    "date": str(row["日期"])[:10],
                    "open": float(row["开盘"]),
                    "high": float(row["最高"]),
                    "low": float(row["最低"]),
                    "close": float(row["收盘"]),
                    "volume": float(row["成交量"]),
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
            latest = df.iloc[0]
            return {
                "pe": self._safe_float(latest.get("市盈率")),
                "pb": self._safe_float(latest.get("市净率")),
                "roe": self._safe_float(latest.get("净资产收益率")),
                "revenue_growth": self._safe_float(latest.get("营业收入同比增长率")),
                "eps": self._safe_float(latest.get("每股收益")),
            }
        except Exception as e:
            raise RuntimeError(f"获取财务数据失败: {e}")

    def get_news(self, symbol: str, limit: int = 5) -> list[dict]:
        try:
            df = ak.stock_news_em(symbol=symbol)
            if df.empty:
                return []
            return [
                {"title": row["标题"], "source": row["文章来源"],
                 "time": str(row["发布时间"]), "url": row["新闻链接"]}
                for _, row in df.head(limit).iterrows()
            ]
        except Exception:
            return []

    def _get_name(self, symbol: str) -> str:
        results = self.search(symbol)
        return results[0]["name"] if results else symbol

    @staticmethod
    def _safe_float(value) -> float | None:
        if value is None or value == "-" or value == "":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
