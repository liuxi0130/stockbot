"""Stock data provider backed by Baostock — free, no registration, server-friendly."""
import urllib.request
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
        self._name_cache: dict[str, str] = {}

    def _lookup_name(self, symbol: str) -> str:
        """Look up actual stock name via Sina API with in-memory cache.

        Falls back to symbol if Sina is unreachable — strictly better than
        the previous behaviour of always returning the code as the name.
        """
        if symbol in self._name_cache:
            return self._name_cache[symbol]
        try:
            sina_code = f"sh{symbol}" if symbol.startswith(("6", "68", "9")) else f"sz{symbol}"
            url = f"https://hq.sinajs.cn/list={sina_code}"
            req = urllib.request.Request(url, headers={"Referer": "https://finance.sina.com.cn"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = resp.read().decode("gbk")
            # Format: var hq_str_sh600519="贵州茅台,1271.18,..."
            if '="' in raw:
                name = raw.split('="')[1].split(",")[0]
                if name and name != symbol:
                    self._name_cache[symbol] = name
                    return name
        except Exception:
            pass
        self._name_cache[symbol] = symbol
        return symbol

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

            # Fallback: if query looks like a stock code but no name match, verify via K-line
            if not rows and query.isdigit() and len(query) == 6:
                try:
                    bs_code = _to_bs_code(query)
                    rs2 = bs.query_history_k_data_plus(
                        bs_code, "date,close",
                        start_date=(datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
                        end_date=datetime.now().strftime("%Y-%m-%d"),
                        frequency="d", adjustflag="2",
                    )
                    if rs2.get_data().empty:
                        return []
                except Exception:
                    return []
                market = "SH" if query.startswith(("6", "68", "9")) else "SZ"
                name = self._lookup_name(query)
                return [{"symbol": query, "name": name, "market": market}]

            return rows[:10]
        except Exception:
            return []

    def get_realtime(self, symbol: str) -> StockQuote:
        """Get latest available daily closing data (T+1 — updated 4–6 PM each trading day).

        Uses adjustflag=2 (不复权) to show actual traded price matching brokerage apps.
        """
        today = datetime.now()
        # Fetch last 10 calendar days — enough to cover weekends/holidays
        start = (today - timedelta(days=10)).strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")
        rs = bs.query_history_k_data_plus(
            _to_bs_code(symbol),
            "date,code,close,preclose,pctChg,volume",
            start_date=start, end_date=end,
            frequency="d", adjustflag="2",
        )
        data = rs.get_data()
        if data.empty:
            raise RuntimeError(f"未获取到 {symbol} 的行情数据")

        latest = data.iloc[-1]
        name = self._lookup_name(symbol)
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
            "date,code,open,high,low,close,volume",
            start_date=start, end_date=end,
            frequency="d", adjustflag="1",
        )

        data = rs.get_data()
        if data.empty:
            raise RuntimeError(f"{symbol} 历史数据为空")

        name = self._lookup_name(symbol)
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
