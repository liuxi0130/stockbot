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
        # Return a mock that has .text attribute
        mock_resp = MagicMock()
        mock_resp.text = "巴西实力强于美国，建议关注让球方。"
        mock_resp.finish_reason = "stop"
        mock_llm.chat.return_value = mock_resp

        advisor = LLMAdvisor(llm=mock_llm)
        matches = [_make_sample_match()]
        results = await advisor.analyze_matches(matches)
        assert len(results) == 1
        assert "巴西" in results[0]

    @pytest.mark.asyncio
    async def test_interpret_strategy_with_mock_llm(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.text = "均衡型适合有经验的彩民。"
        mock_resp.finish_reason = "stop"
        mock_llm.chat.return_value = mock_resp

        advisor = LLMAdvisor(llm=mock_llm)
        strategies = [_make_sample_strategy()]
        result = await advisor.interpret_strategies(strategies, amount=100)
        assert len(result) == 1
        assert result[0].reasoning != "基于规则模型"
