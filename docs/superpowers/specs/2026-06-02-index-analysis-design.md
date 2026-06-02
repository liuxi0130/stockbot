# 上证指数分析功能 — 设计文档

## 概述

新增 `stockbot/index/` 模块和 3 个工具，为 StockBot 补齐大盘层面的分析能力。覆盖上证指数实时行情、技术分析推演、以及结合技术规则与机器学习的多周期预测。

技术规则层纯 Python 计算，ML 层借助 Qlib 框架训练 LightGBM 模型，两者互补输出综合研判。ML 未就绪时降级为纯规则模式。

## 新增模块结构

```
stockbot/
├── index/                          # 新增：指数分析模块
│   ├── __init__.py
│   ├── index_data.py               # IndexDataProvider — 独立的指数数据源
│   ├── index_analyzer.py           # IndexAnalyzer — 指数技术分析（纯规则）
│   └── index_predictor.py          # IndexPredictor — 规则+ML 综合预测
├── tools/
│   ├── market_overview.py          # 新增：get_market_overview 工具
│   ├── index_trend.py              # 新增：analyze_index 工具
│   └── index_predict.py            # 新增：predict_index 工具
└── scripts/
    └── setup_index_model.py        # 新增：训练上证指数 ML 预测模型
```

## 三个工具

| 工具 | 功能 | 核心输出 |
|------|------|---------|
| `get_market_overview` | 大盘行情概况 | 上证指数点位/涨跌幅、涨跌家数、总成交额、行业板块热度前5/后5 |
| `analyze_index` | 指数技术分析 | 多情景推演（乐观/中性/悲观）、关键支撑压力、均线/RSI/MACD/量比 |
| `predict_index` | 多周期指数预测 | 短期信号(1-3天)、中期信号(1-4周)、技术规则信号+ML概率、综合研判 |

## 数据架构

### IndexDataProvider（抽象类）

独立的抽象，不混入个股 `DataProvider`。接口：

```python
class IndexDataProvider(ABC):
    def get_index_quote(self, index_code: str = "000001") -> IndexQuote
    def get_market_breadth(self) -> MarketBreadth
    def get_sector_performance(self, top_n: int = 5) -> list[dict]
    def get_index_history(self, index_code: str, period: str) -> list[dict]
    def get_index_news(self, limit: int = 5) -> list[dict]
```

### AkshareIndexProvider（默认实现）

用 akshare 获取上证指数数据：
- `stock_zh_index_daily_em` → 指数 K 线
- `stock_zh_index_spot_em` → 实时指数行情
- `stock_sector_detail` → 行业板块

### 数据类

```python
@dataclass
class IndexQuote:
    code: str
    name: str
    price: float
    change_pct: float
    change_amt: float
    volume: float
    turnover: float       # 成交额（亿元）
    timestamp: str

@dataclass
class MarketBreadth:
    up_count: int          # 上涨家数
    down_count: int        # 下跌家数
    flat_count: int        # 平盘家数
    total_turnover: float  # 总成交额（亿元）
    limit_up: int          # 涨停家数
    limit_down: int        # 跌停家数
```

## 预测架构

```
IndexPredictor
├── RuleEngine          # 技术指标规则 → 规则信号
│   ├── 均线排列 (5/10/20/60MA)
│   ├── MACD 金叉/死叉
│   ├── RSI 超买超卖 (14日)
│   ├── 量价配合 (量比分析)
│   └── 布林带位置
├── MLPredictor         # Qlib LightGBM → 统计概率（可选）
│   ├── 短期模型 (预测1-3日涨跌方向)
│   └── 中期模型 (预测1-4周涨跌方向)
└── Combiner            # 规则信号 + ML概率 → 综合研判
    ├── 短期信号
    ├── 中期信号
    └── 综合评分 (0-100)
```

### 信号定义

| 评分范围 | 信号 | 含义 |
|---------|------|------|
| 80-100 | 🟢 强烈看多 | 规则+ML 一致看多 |
| 60-79 | 🟡 温和看多 | 多数指标偏多 |
| 40-59 | ⚪ 中性/震荡 | 信号分歧或横盘 |
| 20-39 | 🟠 温和看空 | 多数指标偏空 |
| 0-19 | 🔴 强烈看空 | 规则+ML 一致看空 |

## 与现有系统的集成

### __init__.py 工厂函数

`create_agent()` 中新增 index 工具注册：

```python
index_provider = AkshareIndexProvider()
tool_registry.register(create_market_overview_tool(index_provider))
tool_registry.register(create_index_trend_tool(index_provider))

# ML 模型可选
index_model_dir = cfg.get("index_model", {}).get("model_dir", "data/index_model")
index_predictor = IndexPredictor(
    index_provider=index_provider,
    model_dir=index_model_dir,
    ml_enabled=Path(index_model_dir).exists(),
)
tool_registry.register(create_index_predict_tool(index_predictor))
```

### config.yaml 新增段

```yaml
index:
  default_code: "000001"   # 上证指数
  supported_codes:
    - "000001"              # 上证指数
    - "399001"              # 深证成指
    - "399006"              # 创业板指

index_model:
  model_dir: data/index_model
  short_term_horizon: 3    # 短期预测天数
  mid_term_horizon: 20     # 中期预测天数
```

## 测试策略

### 单元测试文件

```
tests/
├── test_index_data.py       # IndexDataProvider 抽象 + AkshareIndexProvider
├── test_index_analyzer.py   # IndexAnalyzer 纯规则计算
├── test_index_predictor.py  # RuleEngine + Combiner（ML 用 mock）
├── test_market_overview.py  # get_market_overview 工具
├── test_index_trend.py      # analyze_index 工具
└── test_index_predict.py    # predict_index 工具
```

### 测试原则

- `IndexDataProvider` 用 Mock 实现测试，不依赖真实数据源
- 规则引擎纯计算，输入确定输出确定，100% 可测
- ML 部分用 mock predictor 替代，测试 combiner 逻辑
- `is_available()` / `is_ml_available()` 静态方法，测试未就绪降级路径

## 风险提示

所有分析类回复末尾自动附加：

> ⚠️ 以上为基于历史数据和技术指标的多情景推演，不构成投资建议。指数走势受宏观经济、政策变化、国际形势等不可量化因素影响，预测存在不确定性。

## 约束与边界

- 不预测具体点位，只输出方向和概率
- 规则引擎所有计算在 Python 内完成，不依赖 LLM 算数
- ML 为可选增强，未训练时自动降级
- 同一接口可扩展支持深证成指、创业板指等其他指数
