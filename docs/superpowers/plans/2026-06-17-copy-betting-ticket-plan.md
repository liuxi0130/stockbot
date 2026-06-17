# 一键复制体彩推荐单 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在世界杯策略推荐页面的每个投注项旁增加一键复制打票格式文本到剪贴板的按钮

**Architecture:** 纯前端实现。新增 `_format_bet_text()` / `_format_strategy_text()` 纯函数生成打票文本，新增 `_render_copy_button()` 使用 `st.components.v1.html()` 注入带 JS Clipboard API 的 HTML 按钮，在策略渲染的三处位置调用

**Tech Stack:** Python 3.12+, Streamlit, JavaScript (navigator.clipboard API)

---

## 文件变更

| 操作 | 文件 | 说明 |
|------|------|------|
| 修改 | `stockbot/ui/worldcup_page.py` | 新增 3 个辅助函数 + 3 处按钮调用 |
| 新增 | `tests/test_copy_ticket.py` | 格式化函数单元测试 |

---

### Task 1: 编写打票文本格式化函数

**Files:**
- Modify: `stockbot/ui/worldcup_page.py`（在 `_RISK_DESCRIPTIONS` 之后、`render_worldcup()` 之前插入）

- [ ] **Step 1: 添加 `_format_bet_text()` 和 `_format_strategy_text()` 函数**

在 `stockbot/ui/worldcup_page.py` 中，`_RISK_DESCRIPTIONS` 字典之后、`render_worldcup()` 函数之前，插入以下代码：

```python
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
```

- [ ] **Step 2: 验证导入，确保 Bet/Strategy 类型可用**

`worldcup_page.py` 当前不直接 import `Bet`/`Strategy`，但通过 `StrategyEngine.generate()` 返回的 `Strategy` 对象已包含 `Bet` 列表。格式化函数使用 duck typing（通过属性访问），无需显式 import 类型。

- [ ] **Step 3: 提交**

```bash
git add stockbot/ui/worldcup_page.py
git commit -m "feat: add bet text formatting functions for copy feature"
```

---

### Task 2: 编写格式化函数的单元测试

**Files:**
- Create: `tests/test_copy_ticket.py`

- [ ] **Step 1: 创建测试文件并编写测试**

```python
"""Tests for copy-to-clipboard ticket formatting functions."""
import pytest
from stockbot.worldcup.data_provider import Bet, Strategy


class TestFormatBetText:
    """Tests for _format_bet_text()."""

    def test_single_bet_format(self):
        """Single bet should format as: match_id play_type pick ¥stake."""
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
        """Stake should format as integer (round down to nearest 2)."""
        from stockbot.ui.worldcup_page import _format_bet_text

        bet = Bet(
            match_id="周一003",
            home_team="德国",
            away_team="巴西",
            play_type="让球",
            pick="让胜",
            odds=2.10,
            stake=18.0,  # already a multiple of 2
            expected_value=0.22,
            confidence=0.60,
            parlay_legs=[],
        )
        result = _format_bet_text(bet)
        assert result == "周一003 让球 让胜 ¥18"


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
```

- [ ] **Step 2: 运行测试确认失败（函数尚未导入到 worldcup_page）**

```bash
pytest tests/test_copy_ticket.py -v
```

预期：因为 `_format_bet_text` 尚未存在于 `worldcup_page.py`，导入失败（如果 Task 1 已完成则直接通过）。

- [ ] **Step 3: 提交**

```bash
git add tests/test_copy_ticket.py
git commit -m "test: add unit tests for ticket formatting functions"
```

---

### Task 3: 实现 HTML/JS 复制按钮组件

**Files:**
- Modify: `stockbot/ui/worldcup_page.py`（在格式化函数之后插入）

- [ ] **Step 1: 添加 `_render_copy_button()` 函数**

在格式化函数之后、`render_worldcup()` 之前插入：

```python
def _render_copy_button(text: str, key: str = ""):
    """Render a one-click copy-to-clipboard button using JS Clipboard API.

    Args:
        text: The text to copy to clipboard.
        key: Unique key for this button instance (avoids Streamlit duplicate ID warnings).

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
        btn.innerHTML = '✅ 已复制';
        setTimeout(function() {{ btn.innerHTML = '📋 复制'; }}, 1500);
    }}).catch(function() {{
        btn.innerHTML = '❌ 复制失败';
        setTimeout(function() {{ btn.innerHTML = '📋 复制'; }}, 1500);
    }});
">📋 复制</button>
</body>
</html>"""
    st.components.v1.html(html, height=32, scrolling=False)
```

- [ ] **Step 2: 验证 HTML 组件无语法错误**

运行 Streamlit 应用确认页面正常加载（按钮尚未被调用，不影响现有功能）：

```bash
streamlit run app.py --server.headless true --server.port 8501 &
```

确认 worldcup 页面无报错，然后停止。

- [ ] **Step 3: 提交**

```bash
git add stockbot/ui/worldcup_page.py
git commit -m "feat: add HTML/JS copy-to-clipboard button component"
```

---

### Task 4: 在策略卡片中集成复制按钮

**Files:**
- Modify: `stockbot/ui/worldcup_page.py`（`_render_strategy_section()` 函数内）

- [ ] **Step 1: 在串关投注区添加独立复制按钮**

在 `_render_strategy_section()` 中，找到串关渲染代码（`if parlays:` 块内的 `for b in parlays:` 循环），在每个串关的 `st.divider()` 之前插入复制按钮。

定位当前代码（约 line 227-257），修改为：

```python
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
                                # ── Copy button for this parlay ──
                                _render_copy_button(
                                    _format_bet_text(b),
                                    key=f"copy_parlay_{s.risk_level}_{b.play_type}",
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
```

- [ ] **Step 2: 在单关投注区添加独立复制按钮**

找到单关渲染代码（约 line 260-273），修改为：

```python
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
                                    key=f"copy_single_{s.risk_level}_{b.match_id}",
                                )
                                st.divider()
```

- [ ] **Step 3: 在策略卡片顶部添加"复制全部"按钮**

在每个策略卡片的指标区之后、策略说明 expander 之前，添加复制全部按钮。定位当前代码（约 line 212-219，`st.metric` 区域之后），修改为：

```python
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
                        _render_copy_button(
                            strategy_text,
                            key=f"copy_all_{s.risk_level}",
                        )

                    with st.expander("📝 策略说明", expanded=False):
```

- [ ] **Step 4: 手动验证**

启动应用，生成策略，点击各复制按钮确认：
1. 单关复制按钮 → 剪贴板内容正确
2. 串关复制按钮 → 剪贴板内容包含所有腿 + 合并行
3. 复制全部按钮 → 剪贴板内容包含所有投注
4. 按钮反馈 "✅ 已复制" 正常显示并在 1.5 秒后恢复

```bash
streamlit run app.py
```

- [ ] **Step 5: 提交**

```bash
git add stockbot/ui/worldcup_page.py
git commit -m "feat: integrate copy buttons into strategy cards"
```

---

### Task 5: 边界情况处理与最终验证

**Files:**
- Modify: `stockbot/ui/worldcup_page.py`

- [ ] **Step 1: 处理空策略边界情况**

检查 `_format_strategy_text()` 对空 bets 列表的处理 — 返回 `""`，且 `_render_copy_button()` 不会被调用（被 `if strategy_text:` 保护）。

- [ ] **Step 2: 处理特殊字符转义**

验证投注选项名称包含特殊字符（如 `让胜`、`胜平负`）时，JS 模板字符串不出现注入问题。当前转义策略：反斜杠、反引号、美元符号。中文和常见字符不需要转义。

- [ ] **Step 3: 运行全部测试**

```bash
pytest tests/test_copy_ticket.py -v
```

预期：所有测试通过。

- [ ] **Step 4: 提交**

```bash
git add stockbot/ui/worldcup_page.py
git commit -m "chore: finalize copy ticket feature with edge case handling"
```
