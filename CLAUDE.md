# StockBot — AI 驱动的 A 股智能分析助手

> **AI 开发强制规则：修改任何代码前必须先完整阅读本文档。**

## 项目概述

基于 DeepSeek LLM 的 ReAct Agent，提供 A 股行情查询、技术分析、财务解读、新闻搜索、量化预测、大盘分析等功能。支持 CLI (Rich) 和 Web (Streamlit) 两种交互模式。

- **仓库**: https://github.com/liuxi0130/stockbot
- **部署**: 华为云 Flexus (Ubuntu 24.04) + Docker，公网 `123.60.79.29:8501`
- **数据源**: Baostock（主力，免费免注册）→ TuShare（备选，需 Token）→ Akshare（兜底，会被反爬）

## 技术栈

| 层 | 选型 | 版本 |
|------|------|------|
| 语言 | Python | 3.12+ |
| LLM | DeepSeek (OpenAI 兼容) | deepseek-chat |
| 股票数据 | baostock / tushare / akshare | 按优先级降级 |
| Web UI | Streamlit | >=1.28 |
| CLI UI | Rich | >=13.0 |
| 数据库 | SQLite | 内置 |
| 量化 | Qlib + LightGBM | 0.9.7 (当前未就绪) |
| 认证 | bcrypt | >=4.0 |
| 部署 | Docker + docker compose | |

## 完整文件清单与职责

### 顶层
```
config.yaml          — 全局配置 (LLM参数、限额、数据源、端口)
requirements.txt     — Python 依赖
Dockerfile           — Docker 镜像 (Python 3.12-slim + Streamlit)
docker-compose.yml   — 容器编排 (端口8501, 挂载data/, 环境变量)
.env.template        — 环境变量模板 (不上传git)
.gitignore           — 排除 .env, /data/, __pycache__, .streamlit
README.md            — 用户文档
CLAUDE.md            — AI 开发文档 (本文件)
app.py               — Web 入口 (Streamlit)
cli.py               — CLI 入口 (Rich)
```

### stockbot/ — 核心包

#### 工厂与核心
```
__init__.py          — create_agent() 工厂函数，组装所有组件
core.py              — AgentCore: ReAct 推理-工具循环 (max 8轮)
context.py           — ContextAssembler: 拼装 messages (system prompt + 历史 + 画像)
events.py            — StreamEvent 数据类 (TextDelta, ToolCallStart, Error 等)
config.py            — YAML 配置加载器，支持 ${ENV_VAR} 环境变量替换
```

#### LLM 层
```
llm/__init__.py      — 包初始化
llm/base.py          — LLMProvider 抽象类: chat(messages, tools) -> LLMResponse
llm/deepseek.py      — DeepSeekProvider: OpenAI 兼容协议，base_url="https://api.deepseek.com"
```

#### 数据层 (重点)
```
data/__init__.py         — 包初始化
data/base.py             — DataProvider 抽象类 + StockQuote/StockHistory 数据类
                          — 接口: search(), get_realtime(), get_history(), get_financial(), get_news()
data/baostock_provider.py — ⭐ BaostockProvider: 主力数据源，免费免注册，T+1日线
                          — ⚠️ query_history_k_data_plus 禁止传 'name' 字段 (返回空)
                          — ⚠️ query_stock_basic 只按名称搜，纯代码用K线验证兜底
                          — 财务数据用 query_profit_data (字段索引: 0=code,1=pubDate,2=statDate,3=roeAvg,4=npMargin,5=gpMargin,6=netProfit,7=epsTTM,8=MBRevenue)
data/tushare_provider.py — TushareProvider: 需 TUSHARE_TOKEN (付费/积分制)
data/akshare_provider.py — AkshareProvider: 爬虫方案，华为云IP会被反爬，仅兜底
```

#### 工具层
```
tools/__init__.py      — 包初始化
tools/base.py          — Tool 数据类: {name, description, parameters, func}
tools/registry.py      — ToolRegistry: 注册、schema生成、执行调度
tools/stock_search.py  — search_stock: 按名称/代码搜索股票
tools/stock_price.py   — get_realtime_quote: 实时行情 (价格/涨跌/量)
tools/stock_finance.py — get_financial_data: 财务指标 (PE/PB/ROE/EPS)
tools/stock_trend.py   — analyze_trend: 多情景技术分析 (MA/MACD/RSI/量比)
tools/stock_news.py    — search_news: 股票新闻搜索
tools/stock_quant.py   — get_quant_prediction: 量化评分与排名 (需Qlib模型就绪)
tools/market_overview.py — get_market_overview: 大盘概况 (指数/涨跌家数/板块)
tools/index_trend.py   — analyze_index: 指数技术分析
tools/index_predict.py — predict_index: 指数多周期预测 (规则+ML)
```

#### 指数分析模块
```
index/__init__.py       — 包初始化
index/index_data.py     — AkshareIndexProvider: 指数数据源 (上证/深证/创业板)
index/index_analyzer.py — IndexAnalyzer: 纯规则技术分析 (MA/MACD/RSI/布林)
index/index_predictor.py — IndexPredictor: 规则+ML 综合预测 (ML未就绪时降级)
```

#### 量化模块 (当前未激活)
```
quant/__init__.py    — 包初始化
quant/predictor.py   — QuantPredictor: 加载预训练LightGBM模型，CSI300批量预测
                      — ⚠️ Qlib 0.9.x 废弃了 qlib.run.get_data，数据下载方式待解决
```

#### 记忆/存储层
```
memory/__init__.py  — 包初始化
memory/store.py     — MemoryStore: SQLite 底层操作 (init_schema, CRUD)
memory/history.py   — ConversationHistory: 加载最近N轮对话，不截断tool配对
memory/profile.py   — ProfileManager: 用户画像 (自选股/偏好)，显式+隐式学习
```

#### 业务层
```
quota.py   — QuotaManager: 每日每用户5次LLM调用限额，管理员可提额
auth.py    — AuthManager: bcrypt 注册/登录，session管理
admin.py   — AdminService: 用户管理、统计、配置
```

#### UI 层
```
ui/__init__.py    — 包初始化
ui/cli_ui.py      — Rich 终端界面 (顶栏/对话区/状态栏)
ui/login_page.py  — Streamlit 登录/注册页
ui/chat_page.py   — Streamlit 聊天界面
ui/admin_page.py  — Streamlit 管理面板
```

### tests/ — 测试
```
conftest.py              — pytest fixtures (临时db、mock provider)
test_data_provider.py    — DataProvider 接口测试
test_llm_provider.py     — LLM Provider 测试
test_memory_store.py     — SQLite CRUD 测试
test_history.py          — 对话历史管理测试
test_profile.py          — 用户画像测试
test_tool_registry.py    — ToolRegistry 测试
test_stock_tools.py      — 5个股票工具测试
test_quota.py            — 配额管理测试
test_auth.py             — 认证测试
test_context.py          — ContextAssembler 测试
test_agent_core.py       — AgentCore 循环测试
test_quant_tool.py       — 量化工具测试
```

### scripts/ — 辅助脚本
```
setup_qlib.py         — Qlib 数据下载 + LightGBM 训练 (⚠️ Qlib 0.9.x 数据下载已失效)
setup_index_model.py  — 指数预测 ML 模型训练
```

## 架构数据流

```
用户输入 (Web/CLI)
  → QuotaManager.check() → blocked? 拒绝 : 继续
  → ContextAssembler.build() → [system_prompt + history + user_msg]
  → AgentCore.run(messages)
    → LLM.chat() → text response? → yield TextDelta → 结束
                 → tool_calls? → ToolRegistry.execute()
                   → DataProvider.get_xxx()
                   → 结果追加到 messages → 回到 LLM.chat() (最多8轮)
  → MemoryStore.save() → conversations + profile 更新
```

## 配置与密钥

### config.yaml 关键项
```yaml
llm.provider: deepseek          # LLM 提供商
ui.web.port: 8501               # Streamlit 端口
quota.daily_limit: 5            # 每人每日免费次数
data.providers: [akshare]       # 数据源优先级 (实际代码动态选择)
```

### 环境变量 (.env, 不入库)
```
DEEPSEEK_API_KEY=sk-xxx         # 必填: DeepSeek API Key
TUSHARE_TOKEN=xxx               # 可选: TuShare Pro Token
ADMIN_PASS=admin123             # 管理员密码
```

### 数据源选择逻辑 (stockbot/__init__.py line 48-59)
1. 有 TUSHARE_TOKEN → TushareProvider (付费，最稳定)
2. 无 TUSHARE_TOKEN → BaostockProvider (免费，主力)
3. Baostock 初始化失败 → AkshareProvider (兜底，云IP易被反爬)

## 关键注意事项 (踩坑记录)

### Baostock
- ❌ `query_history_k_data_plus` 的 fields 参数包含 `name` 会导致返回空 DataFrame
- ❌ `query_stock_basic` 只能按名称搜索，不能按代码搜索，需代码兜底
- ✅ 财务数据 `query_profit_data` 字段索引: [3]=roe, [6]=netProfit, [7]=eps
- ✅ 代码格式: `sh.600519` (沪市) / `sz.000001` (深市)

### 华为云
- ❌ Akshare 所有实时接口反爬 (东方财富/雪球/新浪封云IP)
- ❌ Docker Hub 被墙，需配置阿里云镜像加速
- ❌ PyPI 被墙，需用阿里云镜像
- ❌ GitHub 被墙，git clone 需 ghproxy 或 wget zip

### Qlib
- ❌ Qlib 0.9.x 删除了 `qlib.run.get_data` 模块
- ❌ 微软 Qlib 数据服务器不可用 (`anon mount` 错误)
- ⚠️ 量化预测功能当前不可用，待解决数据下载方式

### Docker
- ✅ docker-compose.yml 无 `version` 字段 (新版废弃)
- ✅ 环境变量需同时出现在 .env 和 docker-compose.yml 的 environment 段
- ✅ data 目录挂载到宿主机持久化 SQLite

## 开发工作流

### 本地开发
```bash
pip install -r requirements.txt
streamlit run app.py          # Web模式
python cli.py                 # CLI模式
pytest tests/ -v              # 运行测试
```

### 提交与部署
```bash
# 1. 本地修改 + 提交
git add -A
git commit -m "type: description"
git push

# 2. 服务器拉取 + 重建
ssh root@123.60.79.29 "cd /opt/stockbot && git pull && docker compose up -d --build"

# 3. 验证 (注意安全组开放 8501)
curl http://localhost:8501/_stcore/health   # 服务器上
http://123.60.79.29:8501                    # 浏览器
```

### Commit 规范
- `feat:` 新功能
- `fix:` Bug 修复
- `chore:` 配置/依赖/构建
- `docs:` 文档
- Co-Authored-By 行必加

## 已知待办

1. **Qlib 量化预测**: 数据下载方式待解决 (Qlib 0.9.x + 微软数据服务器变更)
2. **指数ML模型**: `setup_index_model.py` 训练脚本未执行
3. **实时行情**: Baostock 为 T+1 数据，非真正实时
4. **新闻功能**: Baostock 无新闻API，Ak 被反爬，新闻工具实际无数据
5. **HTTPS**: 未配置 SSL 证书
6. **域名**: 未绑定自定义域名
