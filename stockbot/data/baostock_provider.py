"""Stock data provider backed by Baostock — free, no registration, server-friendly."""
from datetime import datetime, timedelta
import baostock as bs
from stockbot.data.base import DataProvider, StockQuote, StockHistory


def _to_bs_code(symbol: str) -> str:
    """600519 → sh.600519"""
    if symbol.startswith(("6", "68", "9")):
        return f"sh.{symbol}"
    return f"sz.{symbol}"


class BaostockProvider(DataProvider):
    """A-share data via baostock. No token required, T+1 daily data."""

    def __init__(self):
        bs.login()

    def search(self, query: str) -> list[dict]:
        try:
            rs = bs.query_stock_basic(code_name=query)
            rows = []
            while rs.next():
                row = rs.get_row_data()
                code = row[0]
                name = row[1]
                if code.startswith("sh."):
                    symbol = code[3:]
                    market = "SH"
                else:
                    symbol = code[3:]
                    market = "SZ"
                rows.append({"symbol": symbol, "name": name, "market": market})
            return rows[:10]
        except Exception:
            return []

    def get_realtime(self, symbol: str) -> StockQuote:
        """Get latest available daily data (not true realtime — T+1)."""
        today = datetime.now()
        start = (today - timedelta(days=400)).strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")
        rs = bs.query_history_k_data_plus(
            _to_bs_code(symbol),
            "date,code,name,close,preclose,pctChg,volume",
            start_date=start, end_date=end,
            frequency="d", adjustflag="3",
        )
        data = rs.get_data()
        if data.empty:
            raise RuntimeError(f"未获取到 {symbol} 的行情数据")

        latest = data.iloc[-1]
        name = latest["name"] if latest["name"] else symbol
        return StockQuote(
            symbol=symbol,
            name=name,
            price=float(latest["close"]),
            change_pct=float(latest["pctChg"] or 0),
            volume=float(latest["volume"] or 0),
            timestamp=latest["date"],
        )

    def get_history(self, symbol: str, period: str = "3m") -> StockHistory:
        period_days = {"1m": 30, "3m": 90, "6m": 180, "1y": 250}
        days = period_days.get(period, 90)
        today = datetime.now()
        # Fetch generously, then tail to requested days
        start = (today - timedelta(days=days * 3)).strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")

        rs = bs.query_history_k_data_plus(
            _to_bs_code(symbol),
            "date,code,name,open,high,low,close,volume",
            start_date=start, end_date=end,
            frequency="d", adjustflag="3",
        )

        data = rs.get_data()
        if data.empty:
            raise RuntimeError(f"{symbol} 历史数据为空")

        name = data.iloc[-1]["name"] if data.iloc[-1]["name"] else symbol
        result = [
            {
                "date": row["date"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            for _, row in data.tail(days).iterrows()
        ]
        return StockHistory(symbol=symbol, name=name, data=result)

    def get_financial(self, symbol: str) -> dict:
        """Baostock quarterly profit data — ROE, EPS, net profit."""
        try:
            year = datetime.now().year
            for y in range(year, year - 4, -1):
                for q in ["4", "3", "2", "1"]:
                    rs = bs.query_profit_data(
                        code=_to_bs_code(symbol), year=y, quarter=q
                    )
                    if rs.error_code != "0":
                        continue
                    rows = []
                    while rs.next():
                        rows.append(rs.get_row_data())
                    if rows:
                        r = rows[0]
                        # Field indices: 0=code,1=pubDate,2=statDate,3=roeAvg,
                        # 4=npMargin,5=gpMargin,6=netProfit,7=epsTTM,8=MBRevenue
                        return {
                            "pe": None,    # not available in profit_data
                            "pb": None,    # not available in profit_data
                            "roe": self._safe_float(r[3]) if len(r) > 3 else None,
                            "eps": self._safe_float(r[7]) if len(r) > 7 else None,
                            "revenue_growth": None,
                            "net_profit": self._safe_float(r[6]) if len(r) > 6 else None,
                            "net_profit_growth": None,
                        }
            return {}
        except Exception:
            return {}

    def get_news(self, symbol: str, limit: int = 5) -> list[dict]:
        return []

    @staticmethod
    def _safe_float(value) -> float | None:
        if value is None or value == "" or isinstance(value, bool):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
