"""World Cup betting strategy recommendation page — Streamlit UI."""
import asyncio
import logging
import streamlit as st
from stockbot.worldcup.data_provider import WorldCupDataProvider
from stockbot.worldcup.strategy_engine import StrategyEngine

try:
    from stockbot.worldcup.llm_advisor import LLMAdvisor
except ImportError:
    LLMAdvisor = None  # type: ignore

LOGGER = logging.getLogger(__name__)

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

    # Fetch matches if not loaded (short timeout — external APIs may be blocked)
    if not st.session_state.wc_matches:
        with st.spinner("正在获取今日比赛数据..."):
            provider = WorldCupDataProvider(timeout=5.0)
            try:
                st.session_state.wc_matches = asyncio.run(
                    provider.get_today_matches()
                )
            except Exception as e:
                st.warning(f"数据获取失败（可能被反爬拦截）: {e}")
                LOGGER.warning("Match data fetch failed: %s", e)
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
                if LLMAdvisor is not None and llm is not None:
                    advisor = LLMAdvisor(llm=llm)
                    try:
                        analyses = asyncio.run(
                            advisor.analyze_matches(matches)
                        )
                        st.session_state.wc_analyses = analyses
                    except Exception as e:
                        st.warning(f"AI 分析暂时不可用: {e}")
                        st.session_state.wc_analyses = []
                else:
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
        try:
            with st.spinner("正在计算最优策略..."):
                # Step 1: Rule engine
                engine = StrategyEngine()
                strategies = engine.generate(
                    st.session_state.wc_matches, amount
                )

                # Step 2: LLM interpretation (with timeout protection)
                llm = _get_llm()
                if LLMAdvisor is not None and llm is not None:
                    advisor = LLMAdvisor(llm=llm)
                    try:
                        strategies = asyncio.run(
                            advisor.interpret_strategies(strategies, amount)
                        )
                    except Exception as e:
                        st.warning(f"AI 策略解读暂不可用（将使用规则模型）: {e}")
                        LOGGER.warning("LLM strategy interpretation failed: %s", e)

                st.session_state.wc_strategies = strategies
        except Exception as e:
            st.error(f"策略生成失败: {e}")
            LOGGER.error("Strategy generation crashed: %s", e)
            st.session_state.wc_strategies = []
        finally:
            st.session_state.wc_loading = False
    elif st.session_state.wc_loading and not st.session_state.wc_matches:
        # No match data — reset loading state so UI doesn't hang forever
        st.session_state.wc_loading = False

    strategies = st.session_state.wc_strategies

    if not strategies:
        if not st.session_state.wc_matches:
            st.info("暂无比赛数据，无法生成策略。请检查赛程或稍后再试。")
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
                    # Separate singles from parlays
                    singles = [b for b in s.bets if not b.parlay_legs]
                    parlays = [b for b in s.bets if b.parlay_legs]

                    # Max potential return: sum of stake * odds for all bets
                    max_return = sum(b.stake * b.odds for b in s.bets)

                    st.metric("投入", f"{s.total_stake:.0f} 元")
                    st.metric(
                        "预期净收益",
                        f"{s.expected_return:+.0f} 元",
                    )
                    st.metric("最高可中", f"{max_return:.0f} 元")
                    st.metric("最大亏损", f"{s.max_loss:.0f} 元")

                    with st.expander("📝 策略说明", expanded=False):
                        st.caption(s.reasoning or "基于规则模型生成")

                    with st.expander("📋 下注明细", expanded=False):
                        # ── Parlay bets first (highlighted) ──
                        if parlays:
                            st.markdown("##### 🎯 串关投注")
                            for b in parlays:
                                # Show first 2 teams in summary
                                teams = []
                                for leg in b.parlay_legs[:3]:
                                    teams.append(
                                        f"{leg['home']}vs{leg['away']}"
                                    )
                                team_str = " + ".join(teams)
                                if len(b.parlay_legs) > 3:
                                    team_str += f" 等{len(b.parlay_legs)}场"

                                st.markdown(
                                    f"**{b.play_type}** {team_str}"
                                )
                                st.caption(
                                    f"投 {b.pick} | "
                                    f"组合赔率 **{b.odds:.2f}** | "
                                    f"金额 {b.stake:.0f}元 | "
                                    f"可中 {b.stake * b.odds:.0f}元"
                                )
                                # Expandable legs
                                with st.expander(f"查看{b.play_type}明细",
                                                 expanded=False):
                                    for j, leg in enumerate(b.parlay_legs, 1):
                                        st.caption(
                                            f"  腿{j}: {leg['match_id']} "
                                            f"{leg['home']}vs{leg['away']} "
                                            f"→ {leg['pick']} "
                                            f"(@{leg['odds']:.2f})"
                                        )
                                st.divider()

                        # ── Single bets ──
                        if singles:
                            if parlays:
                                st.markdown("##### 📌 单关投注")
                            for b in singles:
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
