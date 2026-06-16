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
