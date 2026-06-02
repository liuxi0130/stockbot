from stockbot.index.index_data import (
    IndexDataProvider, IndexQuote, MarketBreadth, AkshareIndexProvider,
)

__all__ = [
    "IndexDataProvider", "IndexQuote", "MarketBreadth",
    "AkshareIndexProvider",
]

# Import optional submodules — they are created incrementally in later tasks.
# Each may fail to import until its file is added.
try:
    from stockbot.index.index_analyzer import IndexAnalyzer  # noqa: F811
    __all__.append("IndexAnalyzer")
except ImportError:
    pass

try:
    from stockbot.index.index_predictor import IndexPredictor  # noqa: F811
    __all__.append("IndexPredictor")
except ImportError:
    pass
