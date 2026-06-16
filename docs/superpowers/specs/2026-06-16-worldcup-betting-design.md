# 世界杯体彩购买策略推荐 — 设计文档

> 日期：2026-06-16 | 状态：已批准 | 关联：2026年美加墨世界杯（6月11日开幕）

## 1. 功能概述

在 StockBot Streamlit 应用中新增 **"⚽ 世界杯"** 页面。用户输入投资金额，系统自动拉取当日世界杯比赛及竞彩赔率，基于规则引擎 + LLM 增强分析，输出 **3 档下注策略**（保守 → 进取），覆盖全部竞彩玩法（胜平负、让球胜平负、总进球、半全场、比分）。

## 2. 用户流程

```
进入页面 → 自动加载今日比赛列表 → 输入投资金额 → 点击"生成策略"
→ 规则引擎计算（秒级） + LLM 分析（异步并行）
→ 3 档策略卡片展示 → 点击"查看"展开每档明细
```

## 3. 架构

```
stockbot/worldcup/               # 新增包
├── __init__.py                  # 包初始化
├── data_provider.py             # 数据层：爬取比赛+赔率
├── strategy_engine.py           # 规则引擎：评分+凯利分配
└── llm_advisor.py               # LLM 增强：战力分析+策略解读
stockbot/ui/worldcup_page.py     # Streamlit 页面
app.py                           # 微调：sidebar 导航
```

### 3.1 数据流

```
WorldCupDataProvider.get_today_matches()
  → 竞彩网 JSON API（优先）→ 500.com HTML（兜底）
  → List[Match]

StrategyEngine.generate(matches, amount)
  → 多因素评分 → 凯利公式分配 → 3 档风险裁剪
  → List[Strategy]（每档含 List[Bet]）

LLMAdvisor.analyze(match)        # 并行：每场比赛战力分析
LLMAdvisor.interpret(strategy)   # 并行：每档策略解读
  → 填充 reasoning 字段

UI 渲染
```

## 4. 数据结构

### 4.1 Match（比赛）

```python
@dataclass
class Match:
    match_id: str          # "周一001"
    home_team: str         # 主队中文名
    away_team: str         # 客队中文名
    match_time: str        # "09:00"
    league: str            # "世界杯A组" / "1/8决赛"
    spf_odds: tuple        # 胜平负 (胜, 平, 负)
    rqspf_odds: tuple      # 让球胜平负 (胜, 平, 负)
    handicap: int          # 让球数（0=不让）
    total_goals_odds: dict # {0: 8.5, 1: 4.2, 2: 3.1, ...}
    bq_odds: tuple         # 半全场 9 项赔率
    score_odds: dict       # 部分比分赔率
```

### 4.2 Strategy & Bet（策略 & 下注）

```python
@dataclass
class Strategy:
    risk_level: str        # "保守" / "均衡" / "进取"
    total_stake: float     # 总下注金额
    expected_return: float # 预期回报（加权）
    max_loss: float        # 最大亏损（=total_stake）
    bets: List[Bet]
    reasoning: str         # LLM 生成的理由

@dataclass
class Bet:
    match: Match
    play_type: str         # "胜平负" / "让球" / "总进球" / "半全场" / "比分"
    pick: str              # "胜" / "2-3球" 等
    odds: float
    stake: float           # 下注金额（2元倍数）
    expected_value: float
    confidence: float      # 0-1
```

## 5. 策略引擎算法

### 5.1 多因素评分（每场比赛）

| 因素 | 权重 | 来源 |
|------|------|------|
| 赔率隐含概率 | 40% | 1/赔率 归一化 |
| FIFA 排名差 | 20% | 硬编码排名表 |
| 近期战绩 | 20% | LLM 分析或默认值 |
| 主场优势 | 10% | 美/加/墨 +20% |
| 赔率变化 | 10% | 对比初赔（如有） |

### 5.2 凯利公式

```
f* = (b * p - q) / b
b = odds - 1     # 净赔率
p = 评分推导胜率
q = 1 - p
f* = 建议下注占总资金比例
```

### 5.3 3 档风险裁剪

| | 保守 🛡️ | 均衡 ⚖️ | 进取 🚀 |
|---|---|---|---|
| 凯利系数 | 1/4 Kelly | 1/2 Kelly | 满 Kelly |
| 覆盖玩法 | 仅胜平负 | + 让球 | 全部 5 种 |
| 资金使用率上限 | 50% | 75% | 95% |
| 单注上限 | 10% | 15% | 20% |
| 最低概率阈值 | >0.45 | >0.30 | >0.15 |

## 6. LLM 增强

### 6.1 复用现有 Provider

直接使用 `st.session_state.agent` 中的 `DeepSeekProvider`，不新建连接。

### 6.2 两个 Prompt

**战力分析**（每场比赛，200字）：
```
你是足球分析师。以下是今天的一场比赛：
- 主队：{home}，客队：{away}
- 赛事：{league}
- 赔率：胜{spf[0]}/平{spf[1]}/负{spf[2]}

请简要分析（200字内）：1.实力对比 2.关键看点 3.赛果方向
如无法获取实时信息，基于赔率和排名推断，标注「基于有限数据」。
```

**策略解读**（每档策略，100字）：
```
用户投入{amount}元。以下是{risk_level}型策略：
{bets_summary}
请2-3句话说明：1.核心思路 2.适合人群 3.风险提示
```

### 6.3 降级

- LLM 不可用 → 跳过，标注"基于规则模型"
- 超时 10s → 跳过
- 不影响页面渲染，规则结果直接展示

## 7. UI 设计

### 7.1 页面布局

左侧面板：金额输入 + 生成按钮 + 体彩规则提醒
右侧主体：今日比赛列表 → AI 分析 → 3 档策略卡片

### 7.2 Streamlit 组件

- `st.sidebar` → 金额输入 + 按钮
- `st.dataframe` → 今日比赛一览
- `st.expander` → AI 战力分析 + 策略明细
- `st.columns(3)` / `st.tabs()` → 3 档策略
- `st.metric` → 投入/预期回报/最大亏损

### 7.3 边界状态

| 状态 | 处理 |
|------|------|
| 无比赛日 | "今日无世界杯比赛" + 下一场倒计时 |
| API 拉取失败 | "数据源暂不可用，请稍后再试" |
| 金额为空/0 | 生成按钮置灰 |
| 金额非整数 | 提示并取整 |
| 无正期望比赛 | "今日暂无推荐下注的比赛" |

## 8. app.py 改动

```python
# 常规用户导航从 page = "Chat" 改为：
page = st.sidebar.radio("Navigation", ["💬 Chat", "⚽ 世界杯"])

# 新增路由分支：
elif page == "⚽ 世界杯":
    render_worldcup()
```

## 9. 测试计划

### 单元测试

| 测试 | 覆盖 |
|------|------|
| `test_worldcup_data.py` | Match/Strategy 数据结构，API 解析 mock |
| `test_strategy_engine.py` | 评分模型，凯利计算，3 档裁剪，边界值（金额=0，无比赛） |
| `test_llm_advisor.py` | Prompt 构造，降级逻辑 |

### 集成测试

- 完整流程：mock API → 策略生成 → UI 渲染
- 数据源全故障 → 降级展示

## 10. 文件清单

| 文件 | 操作 | 预估行数 |
|------|------|----------|
| `stockbot/worldcup/__init__.py` | 新增 | ~5 |
| `stockbot/worldcup/data_provider.py` | 新增 | ~200 |
| `stockbot/worldcup/strategy_engine.py` | 新增 | ~250 |
| `stockbot/worldcup/llm_advisor.py` | 新增 | ~100 |
| `stockbot/ui/worldcup_page.py` | 新增 | ~200 |
| `app.py` | 修改 +3行 | — |
| `tests/test_worldcup_data.py` | 新增 | ~80 |
| `tests/test_strategy_engine.py` | 新增 | ~120 |
| `tests/test_llm_advisor.py` | 新增 | ~60 |

总计：新增 ~1000 行，修改 ~3 行。
