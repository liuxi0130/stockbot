"""World Cup match data provider — fetches today's matches and odds from
Chinese sports lottery APIs (sporttery.cn → 500.com fallback)."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Match:
    """A single World Cup match with all available bet types and odds."""
    match_id: str              # "周一001"
    home_team: str             # 主队
    away_team: str             # 客队
    match_time: str            # "09:00"
    league: str                # "世界杯A组"
    spf_odds: tuple[float, float, float]       # 胜平负 (胜, 平, 负)
    rqspf_odds: tuple[float, float, float]     # 让球胜平负
    handicap: int = 0                          # 让球数
    total_goals_odds: dict[str, float] = field(default_factory=dict)
    bq_odds: tuple[float, ...] = ()            # 半全场 9 项
    score_odds: dict[str, float] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return f"{self.match_id} {self.home_team} vs {self.away_team}"


@dataclass
class Bet:
    """A single bet recommendation."""
    match_id: str
    home_team: str
    away_team: str
    play_type: str             # "胜平负" / "让球" / "总进球" / "半全场" / "比分"
    pick: str                  # "胜" / "2-3球"
    odds: float
    stake: float               # 2元倍数
    expected_value: float
    confidence: float          # 0-1


@dataclass
class Strategy:
    """One risk-tier strategy with all bets and metadata."""
    risk_level: str            # "保守" / "均衡" / "进取"
    total_stake: float
    expected_return: float
    max_loss: float
    bets: list[Bet] = field(default_factory=list)
    reasoning: str = ""        # filled by LLM


class MatchFetchError(Exception):
    """Raised when match data cannot be fetched from any source."""
    pass
