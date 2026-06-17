# 一键复制体彩推荐单 — 设计规格

**日期**: 2026-06-17
**状态**: 已确认
**范围**: `stockbot/ui/worldcup_page.py`

## 需求

在世界杯竞彩策略推荐页面，为每个投注推荐增加一键复制打票格式文本到系统剪贴板的功能，方便用户直接粘贴到微信发送给彩票店老板打票。

## 打票文本格式

纯文本，每行一笔投注。格式：`场次 玩法 选项 ¥金额`

**单关示例：**
```
周一001 胜平负 胜 ¥20
```

**串关示例（每腿一行 + 末尾合并行）：**
```
周一001 胜平负 胜
周一002 胜平负 平
======== 2串1 组合赔率 4.25 ¥30
```

## 按钮布局

在每个策略卡片（保守/均衡/进取）的"📋 下注明细"区域内：

1. **串关区** — 每个串关投注右侧放一个"📋 复制"按钮，只复制该串关（所有腿 + 合并行）
2. **单关区** — 每笔单关投注右侧放一个"📋 复制"按钮，只复制该笔
3. **策略顶部（可选增强）** — 一个"📋 复制全部"按钮，复制该档所有投注

按钮点击后显示 "✅ 已复制" 短暂反馈（1.5 秒后恢复）。

## 技术方案

使用 Streamlit `st.components.v1.html()` 注入自定义 HTML/JS 按钮，调用浏览器 `navigator.clipboard.writeText()` API。

### 辅助函数

```python
def _render_copy_button(text: str, label: str = "📋 复制", key: str = ""):
    """Render a copy-to-clipboard button using JS Clipboard API."""
```

使用 `st.columns()` 将按钮内联放置在投注明细旁边，或用 `st.components.v1.html()` 嵌入带 JS 的 HTML 按钮。

### 兼容性

- `navigator.clipboard` 需要 HTTPS 或 localhost 环境
- 在不支持的环境下降级：同时展示一个 `st.code(text)` 块供用户手动选择复制

## 改动范围

**仅修改**: `stockbot/ui/worldcup_page.py`

- 新增 `_render_copy_button(text, label, key)` 辅助函数（约 20 行）
- 在策略渲染的三处位置调用：
  1. 串关投注循环内（line 227-257）
  2. 单关投注循环内（line 263-273）
  3. 策略卡片顶部（可选，复制全部）

## 不做什么

- 不修改数据模型（Match/Bet/Strategy）
- 不修改策略引擎逻辑
- 不增加新依赖包（纯 Streamlit + JS）
- 不增加 CLI 端的复制功能（CLI 没有投注页面）
- 不持久化复制历史
