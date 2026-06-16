"""World Cup betting strategy recommendation module."""
from stockbot.worldcup.data_provider import (
    Match,
    Strategy,
    Bet,
)

try:
    from stockbot.worldcup.data_provider import WorldCupDataProvider
except ImportError:
    WorldCupDataProvider = None  # type: ignore

try:
    from stockbot.worldcup.strategy_engine import StrategyEngine
except ImportError:
    StrategyEngine = None  # type: ignore

try:
    from stockbot.worldcup.llm_advisor import LLMAdvisor
except ImportError:
    LLMAdvisor = None  # type: ignore

__all__ = [
    "Match",
    "Strategy",
    "Bet",
    "WorldCupDataProvider",
    "StrategyEngine",
    "LLMAdvisor",
]
