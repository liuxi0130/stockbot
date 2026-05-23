from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StockQuote:
    symbol: str
    name: str
    price: float
    change_pct: float
    volume: float
    timestamp: str


@dataclass
class StockHistory:
    symbol: str
    name: str
    data: list[dict]


class DataProvider(ABC):
    """Abstract interface for stock data sources."""

    @abstractmethod
    def search(self, query: str) -> list[dict]:
        """Search stocks by name or code. Returns list of {symbol, name, market}."""
        ...

    @abstractmethod
    def get_realtime(self, symbol: str) -> StockQuote:
        """Get real-time quote for a stock symbol."""
        ...

    @abstractmethod
    def get_history(self, symbol: str, period: str) -> StockHistory:
        """Get historical OHLCV data. period: '1m', '3m', '6m', '1y'."""
        ...

    @abstractmethod
    def get_financial(self, symbol: str) -> dict:
        """Get financial indicators: PE, PB, ROE, revenue_growth, etc."""
        ...

    @abstractmethod
    def get_news(self, symbol: str, limit: int = 5) -> list[dict]:
        """Get recent news for a stock."""
        ...
