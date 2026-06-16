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
