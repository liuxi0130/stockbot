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


def _format_bet_text(bet) -> str:
    """Format a single Bet as ticket-printing text.

    Single bet:
        周一001 胜平负 胜 ¥20

    Parlay bet:
        周一001 胜平负 胜
        周一002 让球 平
        ======== 2串1 组合赔率 4.25 ¥30
    """
    lines = []
    if bet.parlay_legs:
        for leg in bet.parlay_legs:
            lines.append(
                f"{leg['match_id']} {leg['play_type']} {leg['pick']}"
            )
        lines.append(
            f"======== {bet.play_type} "
            f"组合赔率 {bet.odds:.2f} "
            f"¥{bet.stake:.0f}"
        )
    else:
        lines.append(
            f"{bet.match_id} {bet.play_type} "
            f"{bet.pick} ¥{bet.stake:.0f}"
        )
    return "\n".join(lines)


def _format_strategy_text(strategy) -> str:
    """Format all bets in a strategy, separated by blank lines."""
    parts = []
    for bet in strategy.bets:
        parts.append(_format_bet_text(bet))
    return "\n\n".join(parts)


def _render_copy_button(text: str):
    """Render a one-click copy-to-clipboard button using JS Clipboard API.

    Args:
        text: The text to copy to clipboard.

    The button is a self-contained HTML component that:
    - Copies ``text`` to clipboard on click via navigator.clipboard.writeText()
    - Shows "✅ 已复制" feedback for 1.5 seconds
    - Falls back to document.execCommand('copy') for older browsers
    - Shows "❌ 复制失败" on error
    """
    # Escape text for safe embedding in JS template literal
    escaped = (
        text.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("$", "\\$")
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {{
        margin: 0; padding: 0;
        display: flex; justify-content: flex-start; align-items: center;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}
    .copy-btn {{
        font-size: 12px; padding: 3px 10px;
        border: 1px solid #d0d5dd; border-radius: 6px;
        background: #ffffff; color: #344054;
        cursor: pointer; white-space: nowrap;
        transition: all 0.15s ease;
    }}
    .copy-btn:hover {{ background: #f5f5f5; border-color: #bbb; }}
    .copy-btn:active {{ background: #e8e8e8; }}
</style>
</head>
<body>
<button class="copy-btn" onclick="
    var text = `{escaped}`;
    var btn = this;
    function fallbackCopy(t) {{
        return new Promise(function(resolve, reject) {{
            var ta = document.createElement('textarea');
            ta.value = t;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.select();
            try {{ document.execCommand('copy'); resolve(); }}
            catch(e) {{ reject(e); }}
            document.body.removeChild(ta);
        }});
    }}
    var copy = (navigator.clipboard && navigator.clipboard.writeText)
        ? navigator.clipboard.writeText.bind(navigator.clipboard)
        : fallbackCopy;
    copy(text).then(function() {{
        btn.textContent = '✅ 已复制';
        setTimeout(function() {{ btn.textContent = '📋 复制'; }}, 1500);
    }}).catch(function() {{
        btn.textContent = '❌ 复制失败';
        setTimeout(function() {{ btn.textContent = '📋 复制'; }}, 1500);
    }});
">📋 复制</button>
</body>
</html>"""
    st.components.v1.html(html, height=32, scrolling=False)


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

                    # ── Copy all button ──
                    strategy_text = _format_strategy_text(s)
                    if strategy_text:
                        _render_copy_button(strategy_text)

                    with st.expander("📝 策略说明", expanded=False):
                        st.caption(s.reasoning or "基于规则模型生成")

                    with st.expander("📋 下注明细", expanded=False):
                        # ── Parlay bets first (highlighted) ──
                        if parlays:
                            st.markdown("##### 🎯 串关投注")
                            for idx, b in enumerate(parlays):
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
                                # ── Copy button for this parlay ──
                                _render_copy_button(
                                    _format_bet_text(b),
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
                                # ── Copy button for this single bet ──
                                _render_copy_button(
                                    _format_bet_text(b),
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
