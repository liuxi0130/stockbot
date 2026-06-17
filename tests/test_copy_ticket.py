"""Tests for copy-to-clipboard ticket formatting functions."""
import pytest
from stockbot.worldcup.data_provider import Bet, Strategy


class TestFormatBetText:
    """Tests for _format_bet_text()."""

    def test_single_bet_format(self):
        """Single bet should format as: match_id play_type pick stake."""
        from stockbot.ui.worldcup_page import _format_bet_text

        bet = Bet(
            match_id="周一001",
            home_team="阿根廷",
            away_team="法国",
            play_type="胜平负",
            pick="胜",
            odds=1.85,
            stake=20.0,
            expected_value=0.15,
            confidence=0.55,
            parlay_legs=[],
        )
        result = _format_bet_text(bet)
        assert result == "周一001 胜平负 胜 ¥20"

    def test_parlay_bet_format(self):
        """Parlay bet should format each leg + summary line."""
        from stockbot.ui.worldcup_page import _format_bet_text

        bet = Bet(
            match_id="",
            home_team="",
            away_team="",
            play_type="2串1",
            pick="阿根廷胜+法国平",
            odds=4.25,
            stake=30.0,
            expected_value=0.35,
            confidence=0.28,
            parlay_legs=[
                {
                    "match_id": "周一001",
                    "home": "阿根廷",
                    "away": "法国",
                    "pick": "胜",
                    "odds": 1.85,
                    "play_type": "胜平负",
                },
                {
                    "match_id": "周一002",
                    "home": "美国",
                    "away": "日本",
                    "pick": "平",
                    "odds": 3.20,
                    "play_type": "胜平负",
                },
            ],
        )
        result = _format_bet_text(bet)
        expected = (
            "周一001 胜平负 胜\n"
            "周一002 胜平负 平\n"
            "======== 2串1 组合赔率 4.25 ¥30"
        )
        assert result == expected

    def test_single_bet_with_fractional_stake(self):
        """Stake should format as integer via :.0f formatting."""
        from stockbot.ui.worldcup_page import _format_bet_text

        bet = Bet(
            match_id="周一003",
            home_team="德国",
            away_team="巴西",
            play_type="让球",
            pick="让胜",
            odds=2.10,
            stake=18.7,  # fractional, :.0f rounds to 19
            expected_value=0.22,
            confidence=0.60,
            parlay_legs=[],
        )
        result = _format_bet_text(bet)
        assert result == "周一003 让球 让胜 ¥19"


class TestFormatStrategyText:
    """Tests for _format_strategy_text()."""

    def test_multiple_bets_separated_by_blank_line(self):
        """Multiple bets should be separated by double newline."""
        from stockbot.ui.worldcup_page import _format_strategy_text

        bets = [
            Bet(
                match_id="周一001",
                home_team="阿根廷",
                away_team="法国",
                play_type="胜平负",
                pick="胜",
                odds=1.85,
                stake=20.0,
                expected_value=0.15,
                confidence=0.55,
                parlay_legs=[],
            ),
            Bet(
                match_id="周一002",
                home_team="美国",
                away_team="日本",
                play_type="胜平负",
                pick="平",
                odds=3.20,
                stake=10.0,
                expected_value=0.12,
                confidence=0.45,
                parlay_legs=[],
            ),
        ]
        strategy = Strategy(
            risk_level="保守",
            total_stake=30.0,
            expected_return=5.0,
            max_loss=30.0,
            bets=bets,
            reasoning="基于规则模型",
        )
        result = _format_strategy_text(strategy)
        expected = "周一001 胜平负 胜 ¥20\n\n周一002 胜平负 平 ¥10"
        assert result == expected

    def test_empty_strategy(self):
        """Empty strategy should return empty string."""
        from stockbot.ui.worldcup_page import _format_strategy_text

        strategy = Strategy(
            risk_level="保守",
            total_stake=0.0,
            expected_return=0.0,
            max_loss=0.0,
            bets=[],
            reasoning="",
        )
        result = _format_strategy_text(strategy)
        assert result == ""
