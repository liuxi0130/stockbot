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
