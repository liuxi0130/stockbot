"""Stock data provider backed by TuShare (tushare.pro)."""
from datetime import datetime, timedelta
import tushare as ts
from stockbot.data.base import DataProvider, StockQuote, StockHistory


# symbol → ts_code mapping
def _to_ts_code(symbol: str) -> str:
    if symbol.startswith(("6", "68", "9")):
        return f"{symbol}.SH"
    return f"{symbol}.SZ"


def _from_ts_code(ts_code: str) -> str:
    return ts_code.split(".")[0]


class TushareProvider(DataProvider):
    """A-share data via tushare.pro. Requires TUSHARE_TOKEN env var."""

    def __init__(self, token: str):
        ts.set_token(token)
        self._api = ts.pro_api()

    def search(self, query: str) -> list[dict]:
        try:
            df = self._api.stock_basic(
                exchange="", list_status="L",
                fields="ts_code,symbol,name,area,industry,market"
            )
        except Exception:
            return []

        if df is None or df.empty:
            return []

        mask = df["name"].str.contains(query) | df["symbol"].str.contains(query)
        results = df[mask].head(10)

        market_map = {"主板": "SH", "创业板": "SZ", "科创板": "SH"}
        return [
            {
                "symbol": row["symbol"],
                "name": row["name"],
                "market": market_map.get(row.get("market", ""), row["ts_code"].split(".")[-1]),
            }
            for _, row in results.iterrows()
        ]

    def get_realtime(self, symbol: str) -> StockQuote:
        try:
            df = self._api.daily(
                ts_code=_to_ts_code(symbol),
                start_date=(datetime.now() - timedelta(days=10)).strftime("%Y%m%d"),
                end_date=datetime.now().strftime("%Y%m%d"),
            )
            if df is None or df.empty:
                raise RuntimeError(f"未获取到 {symbol} 的行情数据")

            latest = df.iloc[0]
            name = self._get_name(symbol)
            return StockQuote(
                symbol=symbol,
                name=name,
                price=float(latest["close"]),
                change_pct=float(latest.get("pct_chg", 0)),
                volume=float(latest.get("vol", 0)),
                timestamp=str(latest["trade_date"]),
            )
        except Exception as e:
            raise RuntimeError(f"获取行情失败: {e}")

    def get_history(self, symbol: str, period: str = "3m") -> StockHistory:
        period_days = {"1m": 30, "3m": 90, "6m": 180, "1y": 250}
        days = period_days.get(period, 90)
        start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
        end = datetime.now().strftime("%Y%m%d")

        try:
            df = self._api.daily(
                ts_code=_to_ts_code(symbol),
                start_date=start,
                end_date=end,
            )
        except Exception as e:
            raise RuntimeError(f"获取历史数据失败: {e}")

        if df is None or df.empty:
            raise RuntimeError(f"{symbol} 历史数据为空")

        name = self._get_name(symbol)
        df = df.sort_values("trade_date")
        data = [
            {
                "date": str(row["trade_date"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["vol"]),
            }
            for _, row in df.tail(days).iterrows()
        ]
        return StockHistory(symbol=symbol, name=name, data=data)

    def get_financial(self, symbol: str) -> dict:
        try:
            ts_code = _to_ts_code(symbol)
            df = self._api.fina_indicator(
                ts_code=ts_code,
                period=(datetime.now() - timedelta(days=400)).strftime("%Y%m%d"),
            )
            if df is None or df.empty:
                return {}

            latest = df.iloc[0]
            return {
                "pe": self._safe_float(latest.get("pe")),
                "pb": self._safe_float(latest.get("pb")),
                "roe": self._safe_float(latest.get("roe")),
                "revenue_growth": self._safe_float(latest.get("or_yoy")),
                "eps": self._safe_float(latest.get("eps")),
                "net_profit": self._safe_float(latest.get("profit_dedt")),
                "net_profit_growth": self._safe_float(latest.get("profit_dedt_yoy")),
            }
        except Exception:
            return {}

    def get_news(self, symbol: str, limit: int = 5) -> list[dict]:
        return []  # TuShare free tier has no news API

    def _get_name(self, symbol: str) -> str:
        results = self.search(symbol)
        return results[0]["name"] if results else symbol

    @staticmethod
    def _safe_float(value) -> float | None:
        if value is None or value == "" or isinstance(value, bool):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
