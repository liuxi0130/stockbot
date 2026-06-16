# 世界杯体彩购买策略推荐 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 StockBot 新增"⚽ 世界杯"页，用户输入金额 → 拉取竞彩比赛赔率 → 规则引擎 + LLM 分析 → 输出 3 档下注策略

**Architecture:** 新增 `stockbot/worldcup/` 包（data_provider → strategy_engine → llm_advisor），新增 UI 页面 `worldcup_page.py`，微调 `app.py` 加导航项。复用现有 `DeepSeekProvider` 做 LLM 增强，复用 Auth 体系做登录校验。

**Tech Stack:** Python 3.12+, Streamlit, httpx（HTTP 请求）, DeepSeek API（复用现有）

---

## File Structure

| File | Operation | Purpose |
|------|-----------|---------|
| `stockbot/worldcup/__init__.py` | Create | Package init, re-exports |
| `stockbot/worldcup/data_provider.py` | Create | Data classes + match fetcher |
| `stockbot/worldcup/strategy_engine.py` | Create | Scoring + Kelly allocation |
| `stockbot/worldcup/llm_advisor.py` | Create | LLM-powered match analysis + strategy interpretation |
| `stockbot/ui/worldcup_page.py` | Create | Streamlit page renderer |
| `app.py` | Modify | Add "⚽ 世界杯" sidebar nav option |
| `tests/test_worldcup_data.py` | Create | Tests for data structures + provider parsing |
| `tests/test_strategy_engine.py` | Create | Tests for scoring, Kelly, risk tiers |
| `tests/test_llm_advisor.py` | Create | Tests for prompt building + degradation |

---

### Task 1: 数据结构与包初始化

**Files:**
- Create: `stockbot/worldcup/__init__.py`
- Create: `stockbot/worldcup/data_provider.py`
- Create: `tests/test_worldcup_data.py`

- [ ] **Step 1: Create `stockbot/worldcup/__init__.py`**

```python
"""World Cup betting strategy recommendation module."""
from stockbot.worldcup.data_provider import (
    Match,
    Strategy,
    Bet,
    WorldCupDataProvider,
)
from stockbot.worldcup.strategy_engine import StrategyEngine
from stockbot.worldcup.llm_advisor import LLMAdvisor

__all__ = [
    "Match",
    "Strategy",
    "Bet",
    "WorldCupDataProvider",
    "StrategyEngine",
    "LLMAdvisor",
]
```

- [ ] **Step 2: Create `stockbot/worldcup/data_provider.py` with data classes**

```python
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
    total_goals_odds: dict[str, float] = field(default_factory=dict)  # {"0":8.5, "1":4.2, ...}
    bq_odds: tuple[float, ...] = ()            # 半全场 9 项
    score_odds: dict[str, float] = field(default_factory=dict)        # 部分比分

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
```

- [ ] **Step 3: Write test for data classes**

Create `tests/test_worldcup_data.py`:

```python
"""Tests for worldcup data structures."""
import pytest
from stockbot.worldcup.data_provider import Match, Bet, Strategy, MatchFetchError


class TestMatch:
    def test_match_display_name(self):
        m = Match(
            match_id="周一001", home_team="美国", away_team="巴西",
            match_time="09:00", league="世界杯A组",
            spf_odds=(2.5, 3.1, 2.8), rqspf_odds=(1.8, 3.3, 3.5),
            handicap=0,
        )
        assert m.display_name == "周一001 美国 vs 巴西"

    def test_match_defaults(self):
        m = Match(
            match_id="周一002", home_team="英格兰", away_team="德国",
            match_time="12:00", league="世界杯B组",
            spf_odds=(2.1, 3.0, 3.8), rqspf_odds=(1.6, 3.5, 4.2),
        )
        assert m.handicap == 0
        assert m.total_goals_odds == {}
        assert m.bq_odds == ()
        assert m.score_odds == {}


class TestBet:
    def test_bet_fields(self):
        b = Bet(
            match_id="周一001", home_team="美国", away_team="巴西",
            play_type="胜平负", pick="胜", odds=2.5,
            stake=10.0, expected_value=0.15, confidence=0.6,
        )
        assert b.stake == 10.0
        assert b.confidence == 0.6


class TestStrategy:
    def test_strategy_defaults(self):
        s = Strategy(
            risk_level="保守",
            total_stake=50.0,
            expected_return=5.0,
            max_loss=50.0,
        )
        assert s.bets == []
        assert s.reasoning == ""

    def test_strategy_with_bets(self):
        b = Bet(
            match_id="周一001", home_team="美国", away_team="巴西",
            play_type="胜平负", pick="胜", odds=2.5,
            stake=10.0, expected_value=0.15, confidence=0.6,
        )
        s = Strategy(
            risk_level="进取", total_stake=95.0,
            expected_return=76.0, max_loss=95.0,
            bets=[b], reasoning="高风险高回报",
        )
        assert len(s.bets) == 1
        assert s.bets[0].pick == "胜"


class TestMatchFetchError:
    def test_error_message(self):
        e = MatchFetchError("API 不可用")
        assert str(e) == "API 不可用"
        assert isinstance(e, Exception)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_worldcup_data.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add stockbot/worldcup/__init__.py stockbot/worldcup/data_provider.py tests/test_worldcup_data.py
git commit -m "feat: add worldcup data structures and package init

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: 策略引擎（评分 + 凯利分配）

**Files:**
- Create: `stockbot/worldcup/strategy_engine.py`
- Create: `tests/test_strategy_engine.py`

- [ ] **Step 1: Write failing test for core scoring function**

Create `tests/test_strategy_engine.py`:

```python
"""Tests for strategy engine — scoring, Kelly, risk tiers."""
import pytest
from stockbot.worldcup.data_provider import Match, Bet, Strategy
from stockbot.worldcup.strategy_engine import StrategyEngine


# ── Sample matches for testing ──

def _make_sample_matches() -> list[Match]:
    return [
        Match(
            match_id="周一001", home_team="美国", away_team="巴西",
            match_time="09:00", league="世界杯A组",
            spf_odds=(2.5, 3.1, 2.8), rqspf_odds=(1.8, 3.3, 3.5),
            handicap=0,
        ),
        Match(
            match_id="周一002", home_team="英格兰", away_team="德国",
            match_time="12:00", league="世界杯B组",
            spf_odds=(2.1, 3.0, 3.8), rqspf_odds=(1.6, 3.5, 4.2),
            handicap=-1,
        ),
        Match(
            match_id="周一003", home_team="法国", away_team="阿根廷",
            match_time="15:00", league="世界杯C组",
            spf_odds=(1.9, 3.2, 4.0), rqspf_odds=(1.5, 3.8, 5.0),
            handicap=0,
        ),
    ]


class TestScoring:
    def test_score_from_odds_returns_0_to_1(self):
        engine = StrategyEngine()
        # 2.5 means implied prob = 1/2.5 = 0.4
        prob = engine._implied_probability(2.5)
        assert 0 < prob < 1
        assert prob == pytest.approx(0.4, 0.01)

    def test_score_from_low_odds_gives_high_prob(self):
        engine = StrategyEngine()
        low_odds_prob = engine._implied_probability(1.5)
        high_odds_prob = engine._implied_probability(5.0)
        assert low_odds_prob > high_odds_prob

    def test_home_advantage_bonus(self):
        engine = StrategyEngine()
        home_bonus = engine._home_advantage("美国")
        neutral = engine._home_advantage("巴西")
        assert home_bonus > neutral

    def test_fifa_rank_score(self):
        engine = StrategyEngine()
        # USA ranked ~16, Brazil ~5 — should favor Brazil
        score = engine._rank_score("美国", "巴西")
        assert -1 <= score <= 1


class TestKellyFormula:
    def test_positive_ev_gives_positive_fraction(self):
        engine = StrategyEngine()
        f = engine._kelly_fraction(odds=2.5, probability=0.5)
        assert f > 0

    def test_negative_ev_gives_zero(self):
        engine = StrategyEngine()
        f = engine._kelly_fraction(odds=2.0, probability=0.4)
        assert f <= 0

    def test_even_odds_fair_prob_gives_zero(self):
        engine = StrategyEngine()
        # odds=2.0, p=0.5 → f* = (1*0.5 - 0.5)/1 = 0
        f = engine._kelly_fraction(odds=2.0, probability=0.5)
        assert f == pytest.approx(0, abs=0.01)


class TestRiskTiers:
    def test_generate_returns_three_strategies(self):
        engine = StrategyEngine()
        matches = _make_sample_matches()
        strategies = engine.generate(matches, amount=100)
        assert len(strategies) == 3
        levels = [s.risk_level for s in strategies]
        assert levels == ["保守", "均衡", "进取"]

    def test_conservative_uses_less_money_than_aggressive(self):
        engine = StrategyEngine()
        matches = _make_sample_matches()
        strategies = engine.generate(matches, amount=100)
        conservative = next(s for s in strategies if s.risk_level == "保守")
        aggressive = next(s for s in strategies if s.risk_level == "进取")
        assert conservative.total_stake <= aggressive.total_stake

    def test_conservative_only_spf_play_type(self):
        engine = StrategyEngine()
        matches = _make_sample_matches()
        strategies = engine.generate(matches, amount=100)
        conservative = next(s for s in strategies if s.risk_level == "保守")
        play_types = {b.play_type for b in conservative.bets}
        assert play_types.issubset({"胜平负"})

    def test_aggressive_covers_more_play_types(self):
        engine = StrategyEngine()
        matches = _make_sample_matches()
        strategies = engine.generate(matches, amount=100)
        aggressive = next(s for s in strategies if s.risk_level == "进取")
        play_types = {b.play_type for b in aggressive.bets}
        # Aggressive should at least have 胜平负 + some others
        assert len(play_types) > 1

    def test_no_matches_returns_empty(self):
        engine = StrategyEngine()
        strategies = engine.generate([], amount=100)
        assert len(strategies) == 3
        for s in strategies:
            assert s.bets == []
            assert s.total_stake == 0

    def test_zero_amount_returns_zero_stakes(self):
        engine = StrategyEngine()
        matches = _make_sample_matches()
        strategies = engine.generate(matches, amount=0)
        for s in strategies:
            assert s.total_stake == 0

    def test_stakes_rounded_to_multiple_of_2(self):
        engine = StrategyEngine()
        matches = _make_sample_matches()
        strategies = engine.generate(matches, amount=100)
        for s in strategies:
            for b in s.bets:
                assert b.stake % 2 == 0, f"Stake {b.stake} is not a multiple of 2"

    def test_no_bet_exceeds_single_bet_cap(self):
        engine = StrategyEngine()
        matches = _make_sample_matches()
        strategies = engine.generate(matches, amount=100)
        conservative = next(s for s in strategies if s.risk_level == "保守")
        for b in conservative.bets:
            assert b.stake <= 10  # 10% of 100
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_strategy_engine.py -v
```

Expected: FAIL — `StrategyEngine` not defined

- [ ] **Step 3: Implement `StrategyEngine`**

Create `stockbot/worldcup/strategy_engine.py`:

```python
"""Strategy engine — multi-factor scoring, Kelly criterion, risk-tier allocation."""
import math
from stockbot.worldcup.data_provider import Match, Bet, Strategy

# Approximate FIFA rankings for 2026 World Cup teams (June 2026 estimates)
_FIFA_RANKING: dict[str, int] = {
    "阿根廷": 1, "法国": 2, "西班牙": 3, "英格兰": 4,
    "巴西": 5, "葡萄牙": 6, "荷兰": 7, "比利时": 8,
    "意大利": 9, "德国": 10, "乌拉圭": 11, "哥伦比亚": 12,
    "墨西哥": 13, "摩洛哥": 14, "日本": 15, "美国": 16,
    "塞内加尔": 17, "伊朗": 18, "丹麦": 19, "韩国": 20,
    "澳大利亚": 23, "加拿大": 27, "沙特": 30, "卡塔尔": 35,
    "新西兰": 40,
}

# 2026 host nations get home-advantage boost
_HOST_NATIONS = {"美国", "加拿大", "墨西哥"}

# Play types available for each risk tier
_SPF_TYPES = ["胜平负"]
_MEDIUM_TYPES = ["胜平负", "让球"]
_ALL_TYPES = ["胜平负", "让球", "总进球", "半全场", "比分"]

# Risk tier configs
_TIER_CONFIG = {
    "保守": {"kelly_factor": 0.25, "max_budget_ratio": 0.50, "max_single": 0.10,
              "min_prob": 0.45, "play_types": _SPF_TYPES},
    "均衡": {"kelly_factor": 0.50, "max_budget_ratio": 0.75, "max_single": 0.15,
              "min_prob": 0.30, "play_types": _MEDIUM_TYPES},
    "进取": {"kelly_factor": 1.00, "max_budget_ratio": 0.95, "max_single": 0.20,
              "min_prob": 0.15, "play_types": _ALL_TYPES},
}


class StrategyEngine:
    """Generate betting strategies from match data and investment amount."""

    # ── Public API ──

    def generate(self, matches: list[Match], amount: float) -> list[Strategy]:
        """Produce 3 risk-tier strategies from today's matches and budget."""
        if not matches or amount <= 0:
            return [
                Strategy(risk_level=tier, total_stake=0,
                         expected_return=0, max_loss=0)
                for tier in ["保守", "均衡", "进取"]
            ]

        strategies = []
        for tier in ["保守", "均衡", "进取"]:
            cfg = _TIER_CONFIG[tier]
            bets = self._generate_bets(matches, amount, cfg)
            total_stake = sum(b.stake for b in bets)
            total_stake = min(total_stake, amount * cfg["max_budget_ratio"])
            # Re-scale if over budget ratio
            if bets and total_stake < sum(b.stake for b in bets):
                scale = total_stake / sum(b.stake for b in bets)
                for b in bets:
                    b.stake = self._round_to_2(b.stake * scale)
            expected_return = sum(b.stake * b.odds * b.confidence for b in bets)
            strategies.append(Strategy(
                risk_level=tier,
                total_stake=total_stake,
                expected_return=round(expected_return - total_stake, 2),
                max_loss=total_stake,
                bets=bets,
            ))
        return strategies

    def _generate_bets(self, matches: list[Match], amount: float,
                       cfg: dict) -> list[Bet]:
        """Generate all candidate bets for a tier, then filter and allocate."""
        candidates: list[tuple[float, Bet]] = []

        for m in matches:
            # ── 胜平负 ──
            for i, pick in enumerate(["胜", "平", "负"]):
                odds = m.spf_odds[i]
                prob = self._match_probability(m, pick, odds)
                kelly = self._kelly_fraction(odds, prob) * cfg["kelly_factor"]
                if kelly > 0 and prob >= cfg["min_prob"]:
                    stake = kelly * amount
                    candidates.append((kelly, Bet(
                        match_id=m.match_id, home_team=m.home_team,
                        away_team=m.away_team,
                        play_type="胜平负", pick=pick, odds=odds,
                        stake=stake,
                        expected_value=self._expected_value(odds, prob),
                        confidence=prob,
                    )))

            # ── 让球 ──
            if "让球" in cfg["play_types"] and m.rqspf_odds != (0, 0, 0):
                for i, pick in enumerate(["胜", "平", "负"]):
                    odds = m.rqspf_odds[i]
                    if odds <= 0:
                        continue
                    prob = self._match_probability(m, pick, odds,
                                                   handicap=m.handicap)
                    kelly = self._kelly_fraction(odds, prob) * cfg["kelly_factor"]
                    if kelly > 0 and prob >= cfg["min_prob"]:
                        stake = kelly * amount
                        candidates.append((kelly, Bet(
                            match_id=m.match_id, home_team=m.home_team,
                            away_team=m.away_team,
                            play_type="让球",
                            pick=f"{pick}({'+' if m.handicap > 0 else ''}{m.handicap})",
                            odds=odds, stake=stake,
                            expected_value=self._expected_value(odds, prob),
                            confidence=prob,
                        )))

            # ── 总进球 ──
            if "总进球" in cfg["play_types"] and m.total_goals_odds:
                best_goals = max(
                    m.total_goals_odds.items(),
                    key=lambda kv: self._implied_probability(float(kv[1]))
                )
                odds = best_goals[1]
                prob = self._implied_probability(odds) * 0.9  # discount for uncertainty
                kelly = self._kelly_fraction(odds, prob) * cfg["kelly_factor"]
                if kelly > 0 and prob >= cfg["min_prob"]:
                    stake = kelly * amount
                    candidates.append((kelly, Bet(
                        match_id=m.match_id, home_team=m.home_team,
                        away_team=m.away_team,
                        play_type="总进球", pick=f"{best_goals[0]}球",
                        odds=odds, stake=stake,
                        expected_value=self._expected_value(odds, prob),
                        confidence=prob,
                    )))

            # ── 半全场 ──
            if "半全场" in cfg["play_types"] and m.bq_odds:
                bq_labels = ["胜胜", "胜平", "胜负", "平胜", "平平", "平负",
                             "负胜", "负平", "负负"]
                if len(m.bq_odds) >= 9:
                    best_idx = max(
                        range(9),
                        key=lambda i: self._implied_probability(m.bq_odds[i])
                    )
                    odds = m.bq_odds[best_idx]
                    prob = self._implied_probability(odds) * 0.85
                    kelly = self._kelly_fraction(odds, prob) * cfg["kelly_factor"]
                    if kelly > 0 and prob >= cfg["min_prob"]:
                        stake = kelly * amount
                        candidates.append((kelly, Bet(
                            match_id=m.match_id, home_team=m.home_team,
                            away_team=m.away_team,
                            play_type="半全场", pick=bq_labels[best_idx],
                            odds=odds, stake=stake,
                            expected_value=self._expected_value(odds, prob),
                            confidence=prob,
                        )))

            # ── 比分 ──
            if "比分" in cfg["play_types"] and m.score_odds:
                best_score = max(
                    m.score_odds.items(),
                    key=lambda kv: self._implied_probability(float(kv[1]))
                )
                odds = best_score[1]
                prob = self._implied_probability(odds) * 0.8
                kelly = self._kelly_fraction(odds, prob) * cfg["kelly_factor"]
                if kelly > 0 and prob >= cfg["min_prob"]:
                    stake = kelly * amount
                    candidates.append((kelly, Bet(
                        match_id=m.match_id, home_team=m.home_team,
                        away_team=m.away_team,
                        play_type="比分", pick=best_score[0],
                        odds=odds, stake=stake,
                        expected_value=self._expected_value(odds, prob),
                        confidence=prob,
                    )))

        # Sort by EV descending, apply single-bet cap, round stakes
        candidates.sort(key=lambda x: x[1].expected_value, reverse=True)
        max_single = amount * cfg["max_single"]
        result = []
        for _, bet in candidates:
            bet.stake = min(self._round_to_2(bet.stake), max_single)
            if bet.stake >= 2:
                result.append(bet)
        return result

    # ── Internal scoring methods ──

    def _match_probability(self, match: Match, pick: str, odds: float,
                           handicap: int = 0) -> float:
        """Compute multi-factor win/draw/loss probability for a pick."""
        implied = self._implied_probability(odds)
        rank_score = self._rank_score(match.home_team, match.away_team)
        home_bonus = self._home_advantage(match.home_team)

        # Adjust base probability by rank difference and home advantage
        if pick in ("胜", "胜胜", "平胜", "负胜") or (
                pick.startswith("胜(") and handicap <= 0):
            adjusted = implied + rank_score * 0.05 + home_bonus * 0.03
        elif pick in ("负", "负负", "胜负", "平负") or (
                pick.startswith("负(") and handicap <= 0):
            adjusted = implied - rank_score * 0.05 - home_bonus * 0.03
        else:
            # Draw — less sensitive to rank diff
            adjusted = implied * (1 - abs(rank_score) * 0.03)

        # Remove margin (overround) — assume ~8% bookmaker margin
        adjusted = adjusted / 1.08
        return max(0.05, min(0.95, adjusted))

    def _implied_probability(self, odds: float) -> float:
        """Convert decimal odds to implied probability."""
        if odds <= 0:
            return 0.0
        return 1.0 / odds

    def _kelly_fraction(self, odds: float, probability: float) -> float:
        """Kelly criterion: f* = (bp - q) / b."""
        b = odds - 1
        if b <= 0:
            return 0.0
        q = 1 - probability
        f = (b * probability - q) / b
        return max(0.0, f)

    def _expected_value(self, odds: float, probability: float) -> float:
        """EV = probability * odds - 1."""
        return probability * odds - 1

    def _rank_score(self, home: str, away: str) -> float:
        """Normalized rank difference score (-1 to 1, positive = home better)."""
        home_rank = _FIFA_RANKING.get(home, 40)
        away_rank = _FIFA_RANKING.get(away, 40)
        diff = away_rank - home_rank
        return max(-1.0, min(1.0, diff / 30.0))

    def _home_advantage(self, team: str) -> float:
        """Host nation bonus (0.0 to 1.0)."""
        return 0.6 if team in _HOST_NATIONS else 0.0

    def _round_to_2(self, value: float) -> float:
        """Round stake down to nearest multiple of 2 (min bet unit)."""
        return max(0, math.floor(value / 2) * 2)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_strategy_engine.py -v
```

Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add stockbot/worldcup/strategy_engine.py tests/test_strategy_engine.py
git commit -m "feat: add strategy engine with scoring and Kelly allocation

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: 数据提供者（比赛拉取 + 解析）

**Files:**
- Modify: `stockbot/worldcup/data_provider.py` (add `WorldCupDataProvider` class)
- Modify: `tests/test_worldcup_data.py` (add provider tests)

- [ ] **Step 1: Write failing tests for WorldCupDataProvider**

Append to `tests/test_worldcup_data.py`:

```python
import json
from unittest.mock import patch, MagicMock
import httpx
from stockbot.worldcup.data_provider import WorldCupDataProvider


# ── Sample sporttery.cn JSON response ──
_SAMPLE_SPORTTERY_RESP = {
    "data": {
        "matchList": [
            {
                "matchId": "周一001",
                "homeTeam": "美国",
                "awayTeam": "巴西",
                "matchTime": "2026-06-17 09:00:00",
                "leagueName": "世界杯A组",
                "odds": {
                    "spf": {"w": 2.50, "d": 3.10, "l": 2.80},
                    "rqspf": {"w": 1.80, "d": 3.30, "l": 3.50, "handicap": 0},
                },
            },
        ],
    },
}


class TestWorldCupDataProvider:
    @pytest.mark.asyncio
    async def test_parse_sporttery_response(self):
        provider = WorldCupDataProvider()
        matches = provider._parse_sporttery_data(_SAMPLE_SPORTTERY_RESP)
        assert len(matches) == 1
        m = matches[0]
        assert m.match_id == "周一001"
        assert m.home_team == "美国"
        assert m.away_team == "巴西"
        assert m.spf_odds == (2.50, 3.10, 2.80)
        assert m.rqspf_odds == (1.80, 3.30, 3.50)
        assert m.handicap == 0
        assert m.league == "世界杯A组"

    @pytest.mark.asyncio
    async def test_get_today_matches_with_mock_api(self):
        provider = WorldCupDataProvider()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _SAMPLE_SPORTTERY_RESP

        with patch.object(provider, "_fetch_json",
                          return_value=_SAMPLE_SPORTTERY_RESP):
            matches = await provider.get_today_matches()
            assert len(matches) == 1

    @pytest.mark.asyncio
    async def test_get_today_matches_empty_when_api_fails(self):
        provider = WorldCupDataProvider()
        with patch.object(provider, "_fetch_json",
                          side_effect=MatchFetchError("网络错误")):
            matches = await provider.get_today_matches()
            assert matches == []

    @pytest.mark.asyncio
    async def test_get_today_matches_fallback_to_500(self):
        provider = WorldCupDataProvider()
        # Simulate sporttery failure, 500 success
        fetch_calls = []

        async def mock_fetch(url, **kwargs):
            fetch_calls.append(url)
            if "sporttery" in url:
                raise MatchFetchError("sporttery down")
            # Mock 500.com response
            return None  # trigger HTML parse path

        with patch.object(provider, "_fetch_html",
                          return_value=None):
            matches = await provider.get_today_matches()
            # Should not crash, empty result for now
            assert isinstance(matches, list)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_worldcup_data.py::TestWorldCupDataProvider -v
```

Expected: FAIL — `WorldCupDataProvider` not defined

- [ ] **Step 3: Implement `WorldCupDataProvider`**

Append to `stockbot/worldcup/data_provider.py`:

```python
"""... (keep existing content above)"""
import logging
import json
import re
import httpx
from datetime import date

LOGGER = logging.getLogger(__name__)

# ── API endpoints ──
_SPORTTERY_URL = "https://webapi.sporttery.cn/gateway/lottery/getFootBallMatchList.qry"
_FIVEHUNDRED_URL = "https://odds.500.com/fenxi/shuju-{date}.shtml"


class WorldCupDataProvider:
    """Fetch today's World Cup matches and odds from Chinese lottery APIs.

    Priority: sporttery.cn (JSON API) → 500.com (HTML parse).
    """

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    async def get_today_matches(self, target_date: str | None = None) -> list[Match]:
        """Fetch all World Cup matches for today (or specified date).

        Returns empty list if no data is available from any source.
        """
        if target_date is None:
            target_date = date.today().strftime("%Y-%m-%d")

        # ── Try sporttery.cn first ──
        try:
            data = await self._fetch_sporttery_games()
            matches = self._parse_sporttery_data(data)
            # Filter to only World Cup matches
            wc_matches = [m for m in matches if "世界杯" in m.league]
            if wc_matches:
                LOGGER.info("Got %d World Cup matches from sporttery.cn",
                            len(wc_matches))
                return wc_matches
        except Exception as e:
            LOGGER.warning("sporttery.cn fetch failed: %s", e)

        # ── Fallback: 500.com ──
        try:
            html = await self._fetch_500com(target_date)
            matches = self._parse_500_html(html)
            if matches:
                LOGGER.info("Got %d matches from 500.com fallback", len(matches))
                return matches
        except Exception as e:
            LOGGER.warning("500.com fetch failed: %s", e)

        LOGGER.warning("No match data available from any source")
        return []

    async def _fetch_sporttery_games(self) -> dict:
        """Fetch match list from sporttery.cn JSON API."""
        params = {
            "matchPage": 1,
            "matchPageSize": 50,
        }
        return await self._fetch_json(_SPORTTERY_URL, params=params)

    async def _fetch_500com(self, target_date: str) -> str:
        """Fetch match page from 500.com."""
        date_str = target_date.replace("-", "")
        url = _FIVEHUNDRED_URL.format(date=date_str)
        return await self._fetch_html(url)

    async def _fetch_json(self, url: str, params: dict | None = None) -> dict:
        """HTTP GET and return JSON dict."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/125.0.0.0 Safari/537.36",
            }
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code != 200:
                raise MatchFetchError(
                    f"HTTP {resp.status_code} from {url}")
            return resp.json()

    async def _fetch_html(self, url: str) -> str:
        """HTTP GET and return HTML text."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36",
            }
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                raise MatchFetchError(
                    f"HTTP {resp.status_code} from {url}")
            return resp.text

    # ── Parsing ──

    def _parse_sporttery_data(self, data: dict) -> list[Match]:
        """Parse sporttery.cn JSON response into Match objects."""
        matches = []
        match_list = data.get("data", {}).get("matchList", [])
        if not match_list:
            # Some versions nest differently
            match_list = data.get("value", {}).get("matchList", [])

        for item in match_list:
            try:
                odds = item.get("odds", {}) or {}
                spf = odds.get("spf", {}) or {}
                rqspf = odds.get("rqspf", {}) or {}

                spf_odds = (
                    float(spf.get("w", 0)),
                    float(spf.get("d", 0)),
                    float(spf.get("l", 0)),
                )
                rqspf_odds = (
                    float(rqspf.get("w", 0)),
                    float(rqspf.get("d", 0)),
                    float(rqspf.get("l", 0)),
                )
                handicap = int(rqspf.get("handicap", 0))

                # Skip if no valid odds
                if spf_odds == (0, 0, 0):
                    continue

                match_time = item.get("matchTime", "")
                if " " in match_time:
                    match_time = match_time.split(" ")[-1][:5]

                m = Match(
                    match_id=item.get("matchId", ""),
                    home_team=item.get("homeTeam", ""),
                    away_team=item.get("awayTeam", ""),
                    match_time=match_time,
                    league=item.get("leagueName", ""),
                    spf_odds=spf_odds,
                    rqspf_odds=rqspf_odds,
                    handicap=handicap,
                )
                matches.append(m)
            except Exception as e:
                LOGGER.debug("Failed to parse match item: %s", e)
                continue

        return matches

    def _parse_500_html(self, html: str) -> list[Match]:
        """Parse 500.com HTML into Match objects (best-effort regex)."""
        if not html:
            return []
        matches = []
        # Extract match rows from the data table
        # 500.com uses table rows with class "bet-tb-tr"
        rows = re.findall(
            r'<tr[^>]*data-match[^>]*>.*?</tr>', html, re.DOTALL
        )
        for row in rows:
            try:
                # Extract match ID
                mid_match = re.search(r'(\w+\d+)', row)
                if not mid_match:
                    continue
                match_id = mid_match.group(1)

                # Extract teams
                teams = re.findall(r'<a[^>]*>([^<]+)</a>', row)
                if len(teams) >= 2:
                    home, away = teams[0], teams[1]
                else:
                    continue

                # Extract SPF odds (first 3 odds after teams)
                odds = re.findall(r'>(\d+\.\d+)<', row)
                spf_odds = tuple(float(o) for o in odds[:3]) if len(odds) >= 3 else (0, 0, 0)

                if spf_odds == (0, 0, 0):
                    continue

                m = Match(
                    match_id=match_id,
                    home_team=home.strip(),
                    away_team=away.strip(),
                    match_time="",
                    league="",
                    spf_odds=spf_odds,
                    rqspf_odds=(0, 0, 0),
                )
                matches.append(m)
            except Exception:
                continue
        return matches
```

- [ ] **Step 4: Add httpx to requirements.txt if not present**

Check and add:

```bash
grep -q "^httpx" requirements.txt || echo "httpx>=0.27.0" >> requirements.txt
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_worldcup_data.py -v
```

Expected: 9 passed (5 from Task 1 + 4 new)

- [ ] **Step 6: Commit**

```bash
git add stockbot/worldcup/data_provider.py tests/test_worldcup_data.py requirements.txt
git commit -m "feat: add WorldCupDataProvider with sporttery.cn + 500.com fallback

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: LLM 增强层

**Files:**
- Create: `stockbot/worldcup/llm_advisor.py`
- Create: `tests/test_llm_advisor.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_llm_advisor.py`:

```python
"""Tests for LLM advisor — prompt building, degradation, response handling."""
import pytest
from unittest.mock import MagicMock, AsyncMock
from stockbot.worldcup.data_provider import Match, Strategy, Bet
from stockbot.worldcup.llm_advisor import LLMAdvisor


def _make_sample_match() -> Match:
    return Match(
        match_id="周一001", home_team="美国", away_team="巴西",
        match_time="09:00", league="世界杯A组",
        spf_odds=(2.5, 3.1, 2.8), rqspf_odds=(1.8, 3.3, 3.5),
        handicap=0,
    )


def _make_sample_strategy() -> Strategy:
    b = Bet(
        match_id="周一001", home_team="美国", away_team="巴西",
        play_type="胜平负", pick="平", odds=3.1,
        stake=10.0, expected_value=0.24, confidence=0.4,
    )
    return Strategy(
        risk_level="均衡", total_stake=30.0,
        expected_return=7.0, max_loss=30.0,
        bets=[b],
    )


class TestLLMAdvisor:
    def test_build_match_analysis_prompt(self):
        advisor = LLMAdvisor(llm=None)
        match = _make_sample_match()
        prompt = advisor._build_match_prompt(match)
        assert "美国" in prompt
        assert "巴西" in prompt
        assert "2.5" in prompt
        assert "世界杯" in prompt

    def test_build_strategy_prompt(self):
        advisor = LLMAdvisor(llm=None)
        strategy = _make_sample_strategy()
        prompt = advisor._build_strategy_prompt(strategy, amount=100)
        assert "100" in prompt
        assert "均衡" in prompt
        assert "巴西" in prompt

    @pytest.mark.asyncio
    async def test_analyze_matches_without_llm_returns_empty(self):
        advisor = LLMAdvisor(llm=None)
        matches = [_make_sample_match()]
        results = await advisor.analyze_matches(matches)
        assert results == []

    @pytest.mark.asyncio
    async def test_interpret_strategies_without_llm_returns_unchanged(self):
        advisor = LLMAdvisor(llm=None)
        strategies = [_make_sample_strategy()]
        result = await advisor.interpret_strategies(strategies, amount=100)
        assert result[0].reasoning == "基于规则模型"

    @pytest.mark.asyncio
    async def test_analyze_match_with_mock_llm(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock()
        mock_llm.chat.return_value.text = "巴西实力强于美国，建议关注让球方。"
        mock_llm.chat.return_value.finish_reason = "stop"

        advisor = LLMAdvisor(llm=mock_llm)
        matches = [_make_sample_match()]
        results = await advisor.analyze_matches(matches)
        assert len(results) == 1
        assert "巴西" in results[0]

    @pytest.mark.asyncio
    async def test_interpret_strategy_with_mock_llm(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock()
        mock_llm.chat.return_value.text = "均衡型适合有经验的彩民。"
        mock_llm.chat.return_value.finish_reason = "stop"

        advisor = LLMAdvisor(llm=mock_llm)
        strategies = [_make_sample_strategy()]
        result = await advisor.interpret_strategies(strategies, amount=100)
        assert len(result) == 1
        assert result[0].reasoning != "基于规则模型"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_llm_advisor.py -v
```

Expected: FAIL — `LLMAdvisor` not defined

- [ ] **Step 3: Implement `LLMAdvisor`**

Create `stockbot/worldcup/llm_advisor.py`:

```python
"""LLM advisor — enriches strategy engine output with AI-powered analysis.

Uses the existing DeepSeekProvider. Degrades gracefully when LLM is unavailable:
- analyze_matches returns [] → UI shows "rule-based only"
- interpret_strategies sets reasoning to "基于规则模型"
"""
import logging
from stockbot.worldcup.data_provider import Match, Strategy

LOGGER = logging.getLogger(__name__)

_MATCH_ANALYSIS_PROMPT = """你是足球分析师。以下是今天的一场比赛：
- 主队：{home}，客队：{away}
- 赛事：{league}
- 赔率：胜{spf_w}/平{spf_d}/负{spf_l}

请简要分析（200字内）：
1. 两队实力对比
2. 关键看点（主场/伤病/战意）
3. 最可能的赛果方向

注意：如无法获取实时信息，请基于赔率和排名做合理推断，在回复中标注「基于有限数据」。
回复格式：直接给出分析内容，不要前导语。"""

_STRATEGY_INTERPRETATION_PROMPT = """你是一位体育彩票策略顾问。用户准备了{amount}元用于世界杯竞彩。

以下是{risk_level}型策略的推荐方案：
{bets_summary}

请用2-3句话（80字内）说明：
1. 这个策略的核心思路
2. 适合什么样的人
3. 主要风险提示

回复格式：直接给出说明，不要前导语。"""


class LLMAdvisor:
    """Enhance strategy output with LLM-powered analysis."""

    def __init__(self, llm=None):
        """llm should be a stockbot.llm.base.LLMProvider instance, or None
        for rule-only mode."""
        self._llm = llm

    async def analyze_matches(self, matches: list[Match]) -> list[str]:
        """Run match-by-match power analysis. Returns list of analysis strings,
        one per match, in the same order as input.

        Returns empty list if LLM is unavailable.
        """
        if self._llm is None:
            return []

        results = []
        for m in matches:
            try:
                prompt = self._build_match_prompt(m)
                messages = [{"role": "user", "content": prompt}]
                resp = await self._llm.chat(messages)
                results.append(resp.text.strip() if resp.text else "")
            except Exception as e:
                LOGGER.warning("LLM match analysis failed for %s: %s",
                               m.match_id, e)
                results.append("")
        return results

    async def interpret_strategies(
        self, strategies: list[Strategy], amount: float
    ) -> list[Strategy]:
        """Add LLM-generated reasoning to each strategy. Modifies in place
        and returns."""
        if self._llm is None:
            for s in strategies:
                s.reasoning = "基于规则模型"
            return strategies

        for s in strategies:
            if not s.bets:
                s.reasoning = "基于规则模型（无符合条件投注）"
                continue
            try:
                summary = self._summarize_bets(s.bets)
                prompt = _STRATEGY_INTERPRETATION_PROMPT.format(
                    amount=int(amount),
                    risk_level=s.risk_level,
                    bets_summary=summary,
                )
                messages = [{"role": "user", "content": prompt}]
                resp = await self._llm.chat(messages)
                s.reasoning = resp.text.strip() if resp.text else "基于规则模型"
            except Exception as e:
                LOGGER.warning("LLM strategy interpretation failed: %s", e)
                s.reasoning = "基于规则模型"

        return strategies

    def _build_match_prompt(self, match: Match) -> str:
        return _MATCH_ANALYSIS_PROMPT.format(
            home=match.home_team,
            away=match.away_team,
            league=match.league,
            spf_w=match.spf_odds[0],
            spf_d=match.spf_odds[1],
            spf_l=match.spf_odds[2],
        )

    def _build_strategy_prompt(
        self, strategy: Strategy, amount: float
    ) -> str:
        summary = self._summarize_bets(strategy.bets)
        return _STRATEGY_INTERPRETATION_PROMPT.format(
            amount=int(amount),
            risk_level=strategy.risk_level,
            bets_summary=summary,
        )

    def _summarize_bets(self, bets: list) -> str:
        """Summarize a list of Bet objects into a compact text."""
        lines = []
        for b in bets[:15]:  # Limit to avoid prompt overflow
            lines.append(
                f"  - {b.match_id} {b.home_team}vs{b.away_team} "
                f"[{b.play_type}] {b.pick} 赔率{b.odds} 投{b.stake:.0f}元"
            )
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_llm_advisor.py -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add stockbot/worldcup/llm_advisor.py tests/test_llm_advisor.py
git commit -m "feat: add LLM advisor for match analysis and strategy interpretation

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Streamlit UI 页面

**Files:**
- Create: `stockbot/ui/worldcup_page.py`

- [ ] **Step 1: Implement UI page**

Create `stockbot/ui/worldcup_page.py`:

```python
"""World Cup betting strategy recommendation page — Streamlit UI."""
import asyncio
import streamlit as st
from stockbot.worldcup.data_provider import WorldCupDataProvider
from stockbot.worldcup.strategy_engine import StrategyEngine
from stockbot.worldcup.llm_advisor import LLMAdvisor

# UI Labels
_RISK_ICONS = {"保守": "🛡️", "均衡": "⚖️", "进取": "🚀"}
_RISK_DESCRIPTIONS = {
    "保守": "低回撤、小额下注、只买胜平负，适合新手",
    "均衡": "中等波动、胜负+让球组合，适合有经验彩民",
    "进取": "高波动、全玩法覆盖，适合追求高回报的玩家",
}


def render_worldcup():
    """Render the World Cup betting strategy page."""
    st.title("⚽ 世界杯竞彩策略推荐")
    st.caption("2026 FIFA World Cup · 美国 | 加拿大 | 墨西哥")

    # ── State initialization ──
    if "wc_matches" not in st.session_state:
        st.session_state.wc_matches = []
    if "wc_strategies" not in st.session_state:
        st.session_state.wc_strategies = []
    if "wc_analyses" not in st.session_state:
        st.session_state.wc_analyses = []
    if "wc_loading" not in st.session_state:
        st.session_state.wc_loading = False

    # ── Sidebar ──
    with st.sidebar:
        st.subheader("💰 投资设置")
        amount = st.number_input(
            "投入金额（元）",
            min_value=0, max_value=100000,
            value=100, step=10,
            help="您计划用于本次竞彩的总资金",
        )

        st.divider()

        can_generate = amount >= 10 and not st.session_state.wc_loading
        if st.button(
            "🎯 生成策略", type="primary",
            use_container_width=True,
            disabled=not can_generate,
        ):
            st.session_state.wc_loading = True
            st.rerun()

        st.divider()
        st.caption("💡 体彩提醒")
        st.caption("• 每注最低 2 元")
        st.caption("• 竞彩截止时间早于开赛 30 分钟")
        st.caption("• 理性购彩，量力而行")

        if st.button("🔄 刷新比赛数据", use_container_width=True):
            st.session_state.wc_matches = []
            st.session_state.wc_strategies = []
            st.session_state.wc_analyses = []
            st.rerun()

    # ── Main content ──
    _render_match_section()
    _render_strategy_section(amount)


def _render_match_section():
    """Render today's match list and AI analyses."""
    st.subheader("📋 今日比赛")

    # Fetch matches if not loaded
    if not st.session_state.wc_matches:
        with st.spinner("正在获取今日比赛数据..."):
            provider = WorldCupDataProvider()
            try:
                st.session_state.wc_matches = asyncio.run(
                    provider.get_today_matches()
                )
            except Exception as e:
                st.error(f"数据获取失败: {e}")
                st.session_state.wc_matches = []

    matches = st.session_state.wc_matches

    if not matches:
        st.info("今日暂无世界杯比赛数据。请检查赛程或稍后再试。")
        return

    # Match overview table
    rows = []
    for m in matches:
        spf_str = f"{m.spf_odds[0]:.2f}/{m.spf_odds[1]:.2f}/{m.spf_odds[2]:.2f}"
        rows.append({
            "编号": m.match_id,
            "主队": m.home_team,
            "客队": m.away_team,
            "时间": m.match_time,
            "赛事": m.league,
            "胜平负赔率": spf_str,
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    # AI analysis expander
    with st.expander("🤖 AI 战力分析", expanded=False):
        if not st.session_state.wc_analyses and st.session_state.wc_loading:
            with st.spinner("AI 正在分析各场比赛..."):
                llm = _get_llm()
                advisor = LLMAdvisor(llm=llm)
                try:
                    analyses = asyncio.run(
                        advisor.analyze_matches(matches)
                    )
                    st.session_state.wc_analyses = analyses
                except Exception as e:
                    st.warning(f"AI 分析暂时不可用: {e}")
                    st.session_state.wc_analyses = []

        analyses = st.session_state.wc_analyses
        if analyses:
            for m, analysis in zip(matches, analyses):
                if analysis:
                    st.markdown(f"**{m.display_name}**")
                    st.caption(analysis)
                    st.divider()
        else:
            st.caption("AI 分析将在生成策略时自动获取，或基于规则模型直接推荐。")


def _render_strategy_section(amount: float):
    """Render the 3-tier strategy output."""
    st.subheader("📊 推荐策略")

    # Generate strategies if triggered
    if st.session_state.wc_loading and st.session_state.wc_matches:
        with st.spinner("正在计算最优策略..."):
            # Step 1: Rule engine
            engine = StrategyEngine()
            strategies = engine.generate(
                st.session_state.wc_matches, amount
            )

            # Step 2: LLM interpretation
            llm = _get_llm()
            advisor = LLMAdvisor(llm=llm)
            try:
                strategies = asyncio.run(
                    advisor.interpret_strategies(strategies, amount)
                )
            except Exception as e:
                st.warning(f"AI 策略解读暂不可用: {e}")

            st.session_state.wc_strategies = strategies
            st.session_state.wc_loading = False

    strategies = st.session_state.wc_strategies

    if not strategies:
        if st.session_state.wc_loading:
            st.info("正在生成策略...")
        else:
            st.info(
                "输入金额并点击 **「🎯 生成策略」** 即可获得推荐。"
            )
        return

    # Render 3 strategy cards in columns
    cols = st.columns(3)

    for i, s in enumerate(strategies):
        icon = _RISK_ICONS.get(s.risk_level, "")
        desc = _RISK_DESCRIPTIONS.get(s.risk_level, "")

        with cols[i]:
            with st.container(border=True):
                st.markdown(f"### {icon} {s.risk_level}型")
                st.caption(desc)

                if s.total_stake > 0:
                    st.metric("投入", f"{s.total_stake:.0f} 元")
                    st.metric(
                        "预期净收益",
                        f"{s.expected_return:+.0f} 元",
                    )
                    st.metric("最大亏损", f"{s.max_loss:.0f} 元")

                    with st.expander("📝 策略说明", expanded=False):
                        st.caption(s.reasoning or "基于规则模型生成")

                    with st.expander("📋 下注明细", expanded=False):
                        for b in s.bets:
                            st.markdown(
                                f"**{b.match_id} {b.home_team}vs{b.away_team}** "
                                f"· {b.play_type}"
                            )
                            st.caption(
                                f"投 {b.pick} | 赔率 {b.odds:.2f} | "
                                f"金额 {b.stake:.0f}元 | "
                                f"信心 {b.confidence:.0%}"
                            )
                            st.divider()
                else:
                    st.caption("本档暂无符合条件的投注")


def _get_llm():
    """Get LLM provider from session state, or None."""
    try:
        agent = st.session_state.get("agent")
        if agent and hasattr(agent, "llm"):
            return agent.llm
    except Exception:
        pass
    return None
```

- [ ] **Step 2: Commit**

```bash
git add stockbot/ui/worldcup_page.py
git commit -m "feat: add World Cup betting strategy Streamlit page

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: 集成到 app.py

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add import and navigation in app.py**

Read `app.py` first to get exact context, then make these two edits.

Edit 1 — Add import at top (after existing `from stockbot.ui.chat_page` line):

In `app.py`, after line 8:
```python
from stockbot.ui.chat_page import render_chat, COOKIE_NAME, _set_session_cookie, _clear_session_cookie
```
Add:
```python
from stockbot.ui.worldcup_page import render_worldcup
```

Edit 2 — Change navigation for regular users. In `app.py`, replace:
```python
    if user.get("role") == "admin":
        page = st.sidebar.radio("Navigation", ["Chat", "Admin"])
    else:
        page = "Chat"

    if page == "Chat":
        render_chat(st.session_state.agent, st.session_state.agent.quota,
                    st.session_state.agent.profile, user)
    elif page == "Admin":
        render_admin(st.session_state.admin_svc, st.session_state.agent.quota)
```

With:
```python
    if user.get("role") == "admin":
        page = st.sidebar.radio("Navigation", ["💬 Chat", "⚽ 世界杯", "🔧 Admin"])
    else:
        page = st.sidebar.radio("Navigation", ["💬 Chat", "⚽ 世界杯"])

    if page == "💬 Chat":
        render_chat(st.session_state.agent, st.session_state.agent.quota,
                    st.session_state.agent.profile, user)
    elif page == "⚽ 世界杯":
        render_worldcup()
    elif page == "🔧 Admin":
        render_admin(st.session_state.admin_svc, st.session_state.agent.quota)
```

- [ ] **Step 2: Verify app starts without import errors**

```bash
python -c "from stockbot.ui.worldcup_page import render_worldcup; print('Import OK')"
```

Expected: `Import OK`

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add World Cup sidebar nav and route to betting page

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: 集成测试与验证

**Files:**
- Modify: `tests/test_worldcup_data.py` (add integration-style test)

- [ ] **Step 1: Add integration test**

Append to `tests/test_worldcup_data.py`:

```python
class TestIntegration:
    """End-to-end pipeline test: data → engine → advisor (no LLM)."""

    def test_full_pipeline_no_llm(self):
        from stockbot.worldcup.data_provider import WorldCupDataProvider
        from stockbot.worldcup.strategy_engine import StrategyEngine
        from stockbot.worldcup.llm_advisor import LLMAdvisor

        # Create a mock match directly (bypass APIs)
        matches = [
            Match(
                match_id="周一001", home_team="美国", away_team="巴西",
                match_time="09:00", league="世界杯A组",
                spf_odds=(2.5, 3.1, 2.8),
                rqspf_odds=(1.8, 3.3, 3.5), handicap=0,
                total_goals_odds={"2": 3.2, "3": 3.8},
                bq_odds=(4.0, 12.0, 25.0, 5.5, 4.5, 6.5, 30.0, 15.0, 8.0),
            ),
            Match(
                match_id="周一002", home_team="英格兰", away_team="德国",
                match_time="12:00", league="世界杯B组",
                spf_odds=(2.1, 3.0, 3.8),
                rqspf_odds=(1.6, 3.5, 4.2), handicap=-1,
            ),
        ]

        # Step 1: Strategy engine
        engine = StrategyEngine()
        strategies = engine.generate(matches, amount=200)

        assert len(strategies) == 3
        conservative = strategies[0]
        assert conservative.risk_level == "保守"
        # Conservative should have bets
        assert len(conservative.bets) > 0
        # All conservative bets are SPF
        assert all(b.play_type == "胜平负" for b in conservative.bets)

        aggressive = strategies[2]
        assert aggressive.risk_level == "进取"
        # Stakes should not exceed budget
        assert conservative.total_stake <= 200
        assert aggressive.total_stake <= 200
        # Conservative should use less than aggressive
        assert conservative.total_stake <= aggressive.total_stake

        # Step 2: LLM advisor (no LLM — degrade path)
        advisor = LLMAdvisor(llm=None)
        result = asyncio.run(
            advisor.interpret_strategies(strategies, amount=200)
        )
        assert len(result) == 3
        for s in result:
            assert "规则" in s.reasoning
```

- [ ] **Step 2: Run all worldcup tests**

```bash
pytest tests/test_worldcup_data.py tests/test_strategy_engine.py tests/test_llm_advisor.py -v
```

Expected: ~28 passed

- [ ] **Step 3: Run full test suite to ensure no regressions**

```bash
pytest tests/ -v --ignore=tests/test_quant_tool.py
```

Expected: All existing tests still pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_worldcup_data.py
git commit -m "test: add integration test for worldcup betting pipeline

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Post-Implementation Verification

- [ ] `streamlit run app.py` — login, verify "⚽ 世界杯" appears in sidebar for regular users
- [ ] Click "⚽ 世界杯" — verify page renders with match list and strategy generation
- [ ] Enter amount and click "🎯 生成策略" — verify 3-tier strategy cards appear
- [ ] Verify degradation when no LLM key available — "基于规则模型" shows
- [ ] Admin user sees all 3 nav items: Chat, 世界杯, Admin
