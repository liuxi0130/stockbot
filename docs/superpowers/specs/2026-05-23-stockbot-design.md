# StockBot — A股智能分析助手 设计文档

## 概述

基于 DeepSeek API 的 AI Agent，提供 A 股实时行情查询、技术分析、财报解读、新闻搜索等服务。CLI 先行，Web 多用户版跟进，本地/云灵活部署。

## 技术栈

| 层 | 选型 | 说明 |
|------|------|------|
| 语言 | Python 3.11+ | |
| LLM | DeepSeek API (OpenAI 兼容) | deepseek-chat 模型 |
| 股票数据 | akshare (默认) | A 股免费数据，provider 模式可扩展 |
| CLI UI | Rich | 终端美化、状态栏、颜色 |
| Web UI | Streamlit | 多用户聊天界面，零命令操作 |
| 数据库 | SQLite | 单文件，免部署 |
| 认证 | bcrypt + session | Web 端多用户隔离 |
| 部署 | Streamlit → Hugging Face / 云服务器 | 三阶段递进 |

## 目录结构

```
stockbot/
├── cli.py                      # CLI 入口
├── app.py                      # Web 入口 (Streamlit)
├── config.yaml                 # 全局配置
├── requirements.txt
├── README.md
├── stockbot/
│   ├── __init__.py
│   ├── core.py                 # AgentCore — 推理-工具循环
│   ├── context.py              # ContextAssembler — 拼装 messages
│   ├── quota.py                # QuotaManager — 多用户用量管控
│   ├── auth.py                 # 注册/登录/鉴权
│   ├── admin.py                # 管理员功能
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py             # LLMProvider 抽象
│   │   └── deepseek.py         # DeepSeek (OpenAI 兼容协议)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py         # ToolRegistry — 注册/匹配/执行
│   │   ├── stock_price.py      # get_realtime_quote
│   │   ├── stock_search.py     # search_stock
│   │   ├── stock_finance.py    # get_financial_data
│   │   ├── stock_trend.py      # analyze_trend (多情景技术分析)
│   │   └── stock_news.py       # search_news
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base.py             # DataProvider 抽象
│   │   └── akshare_provider.py # akshare 实现 (A股)
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── store.py            # SQLite 底层操作
│   │   ├── history.py          # 对话历史管理
│   │   └── profile.py          # 用户画像 (自选股/偏好)
│   └── ui/
│       ├── __init__.py
│       ├── cli_ui.py           # Rich 终端界面
│       ├── login_page.py       # Web 登录/注册页
│       ├── chat_page.py        # Web 聊天界面
│       └── admin_page.py       # Web 管理面板
└── data/
    └── stockbot.db
```

## 架构分层

```
┌──────────────────────────────────────────────┐
│         UI Layer                             │
│  CLI (Rich)  │  Web (Streamlit)              │
│  命令交互     │  多用户 + 认证 + 零命令操作     │
├──────────────────────────────────────────────┤
│         AgentCore                            │
│  ┌─ 推理-工具循环                             │
│  ├─ 消费 ContextAssembler 产出的 messages     │
│  ├─ 调 LLM → 解析响应                         │
│  │  ├─ 文本 → 流式 yield TextDelta            │
│  │  └─ tool_calls → 调 ToolRegistry → 反馈   │
│  └─ 写入 MemoryStore                         │
├──────────┬──────────┬──────────┬─────────────┤
│ LLM      │ Tools    │ Memory   │ Quota       │
│ Provider │ Registry │ Store    │ Manager     │
│          │          │          │             │
│ DeepSeek │ 5 个股票 │ SQLite   │ 用户级限额   │
│ (可扩展) │ 分析工具  │ 多用户   │ 管理员审批   │
└──────────┴──────────┴──────────┴─────────────┘
```

## 核心数据流

```
用户输入
    │
    ▼
ContextAssembler.build(user_input)
    ├─ 系统提示词 (角色 + 工具清单 + 用户画像)
    ├─ 最近 N 轮对话历史 (不截断配对)
    └─ 当前消息
    │
    ▼
QuotaManager.check(user_id) → blocked?
    ├─ 是 → yield QuotaExceeded → return
    └─ 否 → 继续
    │
    ▼
AgentCore.run(messages)
    │
    ┌────┴────┐
    │ DeepSeek │ ← 返回 text 或 tool_calls
    └────┬────┘
         │
    ┌────┴────────────────┐
    │                     │
  文本回复              tool_calls
    │                     │
    ▼                     ▼
yield TextDelta      ToolRegistry.execute()
                        │
                        ▼
                    DataProvider (aksare)
                        │
                        ▼
                    追加 tool_result 到 messages
                        │
                        ▼
                    再次调 LLM (loop, 最多 8 轮)
    │
    ▼
MemoryStore.save()
    ├─ conversations 表写入对话
    └─ profile 表更新用户画像
```

## 模块详细设计

### LLM Provider

- 抽象类 `LLMProvider`，定义 `chat(messages, tools) -> LLMResponse`
- `DeepSeekProvider` 用 `openai` 库，`base_url="https://api.deepseek.com"`
- `LLMResponse` 统一结构: `{text, tool_calls, finish_reason, usage}`
- API Key 从环境变量 `DEEPSEEK_API_KEY` 读取
- 后续加其他模型（OpenAI、本地 Ollama）只需实现抽象类

### AgentCore

- `async run(user_input) -> AsyncIterator[StreamEvent]`
- StreamEvent 类型: `TextDelta | TextDone | ToolCallStart | ToolCallEnd | Error | QuotaExceeded`
- 单循环 (ReAct 模式): think → act → observe → think → ... 
- max_turns=8，防止无限循环
- AgentCore 自身无状态，所有状态在 MemoryStore

### ToolRegistry

- 工具注册、schema 生成、执行调度
- 5 个内置工具，所有工具通过 DataProvider 抽象拿数据

| Tool | 功能 | 参数 |
|------|------|------|
| `search_stock` | 按名称/代码搜索股票 | `query: str` |
| `get_realtime_quote` | 实时行情 (价格/涨跌/量) | `symbol: str` |
| `get_financial_data` | 财报指标 (PE/PB/ROE 等) | `symbol: str`, `metric: str` |
| `analyze_trend` | 多情景技术分析 | `symbol: str`, `period: str` |
| `search_news` | 股票相关新闻 | `symbol: str`, `limit: int` |

- `analyze_trend` 特殊设计: 指标计算完全规则化 (MA/MACD/RSI 数学公式)，LLM 仅负责将结构化数据转化为三种情景叙事 (乐观/中性/悲观)，避免 LLM 幻觉产生虚假价格

### DataProvider

- 抽象类 `DataProvider`，定义 `get_realtime()`, `get_history()`, `search()` 等接口
- `AkshareProvider` 为默认实现 (A 股)
- 后续加 `YfinanceProvider` (美股) 或自定义 provider

### Memory Store

- SQLite 三表: `conversations`, `profile`, `quota`
- `ConversationHistory`: 加载最近 N 轮完整对话，不截断 tool_call/tool_result 配对
- `ProfileManager`: 显式记录 (用户说 "关注茅台") + 隐式学习 (统计查询频率)
- History 超过 200 条自动 trim

### ContextAssembler

- 每次对话前拼装 messages 数组
- System prompt 分四段: 角色 + 工具清单 (动态) + 用户画像 (注入) + 行为规则 (硬约束)
- 所有回复末尾自动附加 "⚠️ 分析仅供参考，不构成投资建议"

### QuotaManager

- 每日每用户免费 5 次 LLM 调用 (管理员可改默认值)
- 计数逻辑: `quota.check()` → blocked? → `quota.consume()` (仅 LLM 调用计数，本地工具不计)
- 管理员可通过面板或 CLI 提额，额度按天重置不累积
- 密码验证: 管理员操作需密码，密码存环境变量 `ADMIN_PASS`

### Auth (Web 专属)

- bcrypt 密码哈希存储
- 登录态用 Streamlit session_state
- 注册默认开放 (可配置关闭)
- 每个用户独立数据空间 (对话/画像/配额)

## UI 设计

### CLI (Rich)

- 顶栏: 应用名 + 版本 + 模型信息
- 对话区: 用户输入白色，Agent 回复绿色流式打印
- 工具执行实时显示: 🔧 黄色 spinner → ✓ 绿色完成
- 底部状态栏常驻: 消息计数 + 数据库路径 + 自选股行情
- 特殊命令: `/help`, `/tools`, `/watch`, `/quota`, `/admin`, `/exit`

### Web (Streamlit)

- 左侧栏: 新对话按钮 + 历史对话列表 + 自选股面板
- 主区域: 聊天界面 + 首次使用快捷示例 + 回复内联操作按钮
- 顶栏图标: 通知/市场概览/自选股/设置/用户菜单
- 快捷卡片: 大盘/热门/自选，点选自动填充问题
- 零命令操作: 所有功能通过点击或自然语言完成
- 管理面板: 用户管理/统计/配置/日志 四 tab

## 部署方案

| 阶段 | 方案 | 命令 | 适用场景 |
|------|------|------|---------|
| 开发 | 本机 | `streamlit run app.py` | 开发调试 |
| 测试 | Hugging Face Spaces | git push 自动部署 | 免费分享给朋友试用 |
| 正式 | 云服务器 + Nginx | systemd 托管 | 正式运营，24×7 在线 |

## 数据模型

```sql
CREATE TABLE users (
    id         TEXT PRIMARY KEY,
    username   TEXT UNIQUE NOT NULL,
    password   TEXT NOT NULL,          -- bcrypt hash
    role       TEXT DEFAULT 'user',    -- user | admin
    daily_quota INTEGER DEFAULT 5,     -- 个人日配额
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE conversations (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id),
    role       TEXT NOT NULL,          -- user | assistant | tool
    content    TEXT NOT NULL,
    tool_name  TEXT,                   -- tool 消息时填充
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE profile (
    user_id    TEXT PRIMARY KEY REFERENCES users(id),
    value      TEXT NOT NULL,          -- JSON
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE quota (
    user_id    TEXT NOT NULL REFERENCES users(id),
    date       TEXT NOT NULL,          -- "2026-05-23"
    calls      INTEGER DEFAULT 0,
    approved   INTEGER DEFAULT 0,     -- 管理员额外批准
    PRIMARY KEY (user_id, date)
);
```

## 约束与边界

- `analyze_trend` 输出多情景推演，不给出确定预测
- 所有分析类回复末尾附加风险提示
- 不提供买卖建议，不推荐具体操作
- 技术指标计算规则化，LLM 不参与数值计算
- 工具调用超时 10s，异常转为友好错误交给 LLM 自行修正
