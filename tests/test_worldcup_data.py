"""Tests for worldcup data structures."""
import json
from unittest.mock import patch, MagicMock
import pytest
import httpx
from stockbot.worldcup.data_provider import (
    Match, Bet, Strategy, MatchFetchError, WorldCupDataProvider,
)


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
    def test_parse_sporttery_response(self):
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
        with patch.object(provider, "_fetch_json",
                          side_effect=MatchFetchError("sporttery down")):
            with patch.object(provider, "_fetch_html",
                              return_value=None):
                matches = await provider.get_today_matches()
                # Should not crash, empty result for now
                assert isinstance(matches, list)


class TestIntegration:
    """End-to-end pipeline test: data → engine → advisor (no LLM)."""

    def test_full_pipeline_no_llm(self):
        import asyncio
        from stockbot.worldcup.strategy_engine import StrategyEngine
        from stockbot.worldcup.llm_advisor import LLMAdvisor

        # Create mock matches directly (bypass APIs)
        matches = [
            Match(
                match_id="周一001", home_team="巴西", away_team="新西兰",
                match_time="09:00", league="世界杯A组",
                spf_odds=(1.3, 5.0, 9.0),
                rqspf_odds=(1.1, 4.5, 7.0), handicap=-2,
                total_goals_odds={"2": 3.2, "3": 3.8},
                bq_odds=(2.0, 12.0, 25.0, 3.5, 4.5, 6.5, 30.0, 15.0, 8.0),
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
        assert len(conservative.bets) > 0
        assert all(b.play_type == "胜平负" for b in conservative.bets)

        aggressive = strategies[2]
        assert aggressive.risk_level == "进取"
        assert conservative.total_stake <= 200
        assert aggressive.total_stake <= 200
        assert conservative.total_stake <= aggressive.total_stake

        # Step 2: LLM advisor (no LLM — degrade path)
        advisor = LLMAdvisor(llm=None)
        result = asyncio.run(
            advisor.interpret_strategies(strategies, amount=200)
        )
        assert len(result) == 3
        for s in result:
            assert "规则" in s.reasoning
