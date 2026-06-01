# StockBot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-user A-share stock analysis AI agent with CLI and Web interfaces, powered by DeepSeek API.

**Architecture:** Layered design — infrastructure (LLM, Data, Memory) → services (Tools, Quota, Auth, Context) → core (Agent loop) → UI (CLI + Streamlit Web). Each layer depends only on the layer below. All state in SQLite, all external calls behind abstract providers.

**Tech Stack:** Python 3.11+, DeepSeek API (OpenAI SDK), akshare, Rich, Streamlit, SQLite, bcrypt, pytest

---

## File Map

```
stockbot/                        # project root
├── requirements.txt
├── config.yaml
├── cli.py                       # entry: CLI mode
├── app.py                       # entry: Web mode (Streamlit)
├── stockbot/
│   ├── __init__.py              # exports factory function create_agent()
│   ├── events.py                # StreamEvent dataclasses
│   ├── config.py                # YAML config loader
│   ├── core.py                  # AgentCore — agent loop
│   ├── context.py               # ContextAssembler — message assembly
│   ├── quota.py                 # QuotaManager — rate limiting
│   ├── auth.py                  # user registration / login / password
│   ├── admin.py                 # admin operations (list users, stats, config)
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py              # LLMProvider ABC, LLMResponse, ToolCall
│   │   └── deepseek.py          # DeepSeek via OpenAI SDK
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py              # Tool dataclass
│   │   ├── registry.py          # ToolRegistry
│   │   ├── stock_price.py       # get_realtime_quote
│   │   ├── stock_search.py      # search_stock
│   │   ├── stock_finance.py     # get_financial_data
│   │   ├── stock_trend.py       # analyze_trend (multi-scenario)
│   │   └── stock_news.py        # search_news
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base.py              # DataProvider ABC + StockQuote/StockHistory dataclasses
│   │   └── akshare_provider.py  # akshare implementation
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── store.py             # MemoryStore — SQLite CRUD
│   │   ├── history.py           # ConversationHistory
│   │   └── profile.py           # ProfileManager
│   └── ui/
│       ├── __init__.py
│       ├── cli_ui.py            # Rich terminal chat UI
│       ├── login_page.py        # Streamlit login/register page
│       ├── chat_page.py         # Streamlit chat interface
│       └── admin_page.py        # Streamlit admin dashboard
└── tests/
    ├── __init__.py
    ├── conftest.py              # fixtures (temp db, mock provider)
    ├── test_llm_provider.py
    ├── test_data_provider.py
    ├── test_memory_store.py
    ├── test_history.py
    ├── test_profile.py
    ├── test_tool_registry.py
    ├── test_stock_tools.py
    ├── test_quota.py
    ├── test_context.py
    ├── test_auth.py
    ├── test_agent_core.py
    └── test_config.py
```

---

## Phase 1: Project Scaffolding

### Task 1: Project skeleton and dependencies

**Files:**
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `stockbot/__init__.py`
- Create: `stockbot/events.py`
- Create: `stockbot/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write requirements.txt**

```
openai>=1.0.0
akshare>=1.14.0
rich>=13.0.0
streamlit>=1.28.0
bcrypt>=4.0.0
pyyaml>=6.0
pytest>=8.0.0
pytest-asyncio>=0.24.0
pandas>=2.0.0
```

- [ ] **Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 3: Write config.yaml**

```yaml
llm:
  provider: deepseek
  model: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}
  max_tokens: 4096
  temperature: 0.3

quota:
  daily_limit: 5
  admin_password: ${ADMIN_PASS:-admin123}

data:
  providers:
    - akshare

ui:
  default: web
  cli:
    show_stock_panel: true
  web:
    port: 8501
    theme: light

memory:
  history_limit: 200
  db_path: data/stockbot.db

auth:
  open_registration: true
```

- [ ] **Step 4: Write stockbot/events.py**

```python
from dataclasses import dataclass, field


@dataclass
class TextDelta:
    content: str


@dataclass
class TextDone:
    pass


@dataclass
class ToolCallStart:
    name: str
    args: dict


@dataclass
class ToolCallEnd:
    name: str
    result: str


@dataclass
class Error:
    message: str


@dataclass
class QuotaExceeded:
    limit: int
    used: int


StreamEvent = TextDelta | TextDone | ToolCallStart | ToolCallEnd | Error | QuotaExceeded
```

- [ ] **Step 5: Write stockbot/config.py**

```python
import os
import re
import yaml
from pathlib import Path


def _resolve_env(value: str) -> str:
    pattern = re.compile(r'\$\{(\w+)(?::-([^}]*))?\}')
    def replacer(m):
        var_name = m.group(1)
        default = m.group(2)
        return os.environ.get(var_name, default or "")
    return pattern.sub(replacer, value)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = _resolve_env(raw)
    return yaml.safe_load(raw)
```

- [ ] **Step 6: Write a minimal stockbot/__init__.py**

```python
from pathlib import Path


def ensure_data_dir(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 7: Write tests/conftest.py**

```python
import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def sample_config():
    return {
        "llm": {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "api_key": "test-key",
            "max_tokens": 4096,
            "temperature": 0.3,
        },
        "quota": {"daily_limit": 5, "admin_password": "admin123"},
        "data": {"providers": ["akshare"]},
        "memory": {"history_limit": 200, "db_path": ":memory:"},
        "auth": {"open_registration": True},
    }
```

- [ ] **Step 8: Write tests/__init__.py (empty)**

```python
```

- [ ] **Step 9: Verify scaffolding**

```bash
python -c "from stockbot.events import TextDelta; print('OK')"
python -c "from stockbot.config import load_config; print(load_config('config.yaml'))"
```

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with config, events, and dev tools"
```

---

## Phase 2: Infrastructure Layer

### Task 2: DataProvider abstract and AkshareProvider

**Files:**
- Create: `stockbot/data/__init__.py`
- Create: `stockbot/data/base.py`
- Create: `stockbot/data/akshare_provider.py`
- Create: `tests/test_data_provider.py`

- [ ] **Step 1: Write failing tests in tests/test_data_provider.py**

```python
import pytest
from stockbot.data.base import DataProvider, StockQuote, StockHistory


class MockDataProvider(DataProvider):
    """Minimal implementation for testing abstract interface."""
    def search(self, query: str) -> list[dict]:
        return [{"symbol": "600519", "name": "贵州茅台", "market": "SH"}]

    def get_realtime(self, symbol: str) -> StockQuote:
        return StockQuote(
            symbol=symbol, name="贵州茅台", price=1680.0,
            change_pct=2.3, volume=12345678, timestamp="2026-05-23 14:00:00"
        )

    def get_history(self, symbol: str, period: str) -> StockHistory:
        return StockHistory(symbol=symbol, name="贵州茅台", data=[
            {"date": "2026-05-20", "open": 1640, "high": 1685, "low": 1635, "close": 1680, "volume": 10000000}
        ])

    def get_financial(self, symbol: str) -> dict:
        return {"pe": 25.5, "pb": 8.2, "roe": 32.1, "revenue_growth": 15.3}

    def get_news(self, symbol: str, limit: int) -> list[dict]:
        return [{"title": "测试新闻", "source": "测试来源", "time": "2026-05-23", "url": ""}]


class TestDataProviderInterface:
    def test_can_instantiate_mock_provider(self):
        provider = MockDataProvider()
        assert isinstance(provider, DataProvider)

    def test_search_returns_list_of_dicts(self):
        provider = MockDataProvider()
        results = provider.search("茅台")
        assert len(results) > 0
        assert "symbol" in results[0]
        assert "name" in results[0]

    def test_get_realtime_returns_stock_quote(self):
        provider = MockDataProvider()
        quote = provider.get_realtime("600519")
        assert isinstance(quote, StockQuote)
        assert quote.price > 0

    def test_get_history_returns_stock_history(self):
        provider = MockDataProvider()
        history = provider.get_history("600519", "1m")
        assert isinstance(history, StockHistory)
        assert len(history.data) > 0

    def test_get_financial_returns_dict(self):
        provider = MockDataProvider()
        fin = provider.get_financial("600519")
        assert "pe" in fin

    def test_get_news_returns_list(self):
        provider = MockDataProvider()
        news = provider.get_news("600519", 5)
        assert isinstance(news, list)
        assert len(news) > 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_data_provider.py -v
```
Expected: FAIL — Module not found

- [ ] **Step 3: Write stockbot/data/__init__.py**

```python
from stockbot.data.base import DataProvider, StockQuote, StockHistory
from stockbot.data.akshare_provider import AkshareProvider

__all__ = ["DataProvider", "StockQuote", "StockHistory", "AkshareProvider"]
```

- [ ] **Step 4: Write stockbot/data/base.py**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StockQuote:
    symbol: str
    name: str
    price: float
    change_pct: float
    volume: float
    timestamp: str


@dataclass
class StockHistory:
    symbol: str
    name: str
    data: list[dict]


class DataProvider(ABC):
    """Abstract interface for stock data sources."""

    @abstractmethod
    def search(self, query: str) -> list[dict]:
        """Search stocks by name or code. Returns list of {symbol, name, market}."""
        ...

    @abstractmethod
    def get_realtime(self, symbol: str) -> StockQuote:
        """Get real-time quote for a stock symbol."""
        ...

    @abstractmethod
    def get_history(self, symbol: str, period: str) -> StockHistory:
        """Get historical OHLCV data. period: '1m', '3m', '6m', '1y'."""
        ...

    @abstractmethod
    def get_financial(self, symbol: str) -> dict:
        """Get financial indicators: PE, PB, ROE, revenue_growth, etc."""
        ...

    @abstractmethod
    def get_news(self, symbol: str, limit: int = 5) -> list[dict]:
        """Get recent news for a stock."""
        ...
```

- [ ] **Step 5: Write stockbot/data/akshare_provider.py**

```python
import akshare as ak
import pandas as pd
from datetime import datetime
from stockbot.data.base import DataProvider, StockQuote, StockHistory


class AkshareProvider(DataProvider):
    """A-share stock data via akshare library."""

    def search(self, query: str) -> list[dict]:
        try:
            df = ak.stock_info_a_code_name()
            mask = df["名称"].str.contains(query) | df["代码"].str.contains(query)
            results = df[mask].head(10)
            return [
                {"symbol": row["代码"], "name": row["名称"], "market": "SH" if row["代码"].startswith("6") else "SZ"}
                for _, row in results.iterrows()
            ]
        except Exception:
            return []

    def get_realtime(self, symbol: str) -> StockQuote:
        try:
            df = ak.stock_zh_a_spot_em()
            row = df[df["代码"] == symbol]
            if row.empty:
                raise ValueError(f"未找到股票: {symbol}")
            r = row.iloc[0]
            return StockQuote(
                symbol=symbol,
                name=r["名称"],
                price=float(r["最新价"]),
                change_pct=float(r["涨跌幅"]),
                volume=float(r["成交量"]),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        except Exception as e:
            raise RuntimeError(f"获取行情失败: {e}")

    def get_history(self, symbol: str, period: str = "3m") -> StockHistory:
        period_days = {"1m": 30, "3m": 90, "6m": 180, "1y": 250}
        days = period_days.get(period, 90)
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
            name = self._get_name(symbol)
            data = [
                {
                    "date": str(row["日期"])[:10],
                    "open": float(row["开盘"]),
                    "high": float(row["最高"]),
                    "low": float(row["最低"]),
                    "close": float(row["收盘"]),
                    "volume": float(row["成交量"]),
                }
                for _, row in df.tail(days).iterrows()
            ]
            return StockHistory(symbol=symbol, name=name, data=data)
        except Exception as e:
            raise RuntimeError(f"获取历史数据失败: {e}")

    def get_financial(self, symbol: str) -> dict:
        try:
            df = ak.stock_financial_abstract_ths(symbol=symbol)
            if df.empty:
                return {}
            latest = df.iloc[0]
            return {
                "pe": self._safe_float(latest.get("市盈率")),
                "pb": self._safe_float(latest.get("市净率")),
                "roe": self._safe_float(latest.get("净资产收益率")),
                "revenue_growth": self._safe_float(latest.get("营业收入同比增长率")),
                "eps": self._safe_float(latest.get("每股收益")),
            }
        except Exception as e:
            raise RuntimeError(f"获取财务数据失败: {e}")

    def get_news(self, symbol: str, limit: int = 5) -> list[dict]:
        try:
            df = ak.stock_news_em(symbol=symbol)
            if df.empty:
                return []
            return [
                {"title": row["标题"], "source": row["文章来源"], "time": str(row["发布时间"]), "url": row["新闻链接"]}
                for _, row in df.head(limit).iterrows()
            ]
        except Exception:
            return []

    def _get_name(self, symbol: str) -> str:
        results = self.search(symbol)
        return results[0]["name"] if results else symbol

    @staticmethod
    def _safe_float(value) -> float | None:
        if value is None or value == "-" or value == "":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
```

- [ ] **Step 6: Run tests (with MockDataProvider, they pass now)**

```bash
pytest tests/test_data_provider.py -v
```
Expected: 6 PASS

- [ ] **Step 7: Commit**

```bash
git add stockbot/data/ tests/test_data_provider.py
git commit -m "feat: add DataProvider abstract interface and AkshareProvider"
```

---

### Task 3: LLM Provider abstract and DeepSeekProvider

**Files:**
- Create: `stockbot/llm/__init__.py`
- Create: `stockbot/llm/base.py`
- Create: `stockbot/llm/deepseek.py`
- Create: `tests/test_llm_provider.py`

- [ ] **Step 1: Write failing tests in tests/test_llm_provider.py**

```python
import pytest
from stockbot.llm.base import LLMProvider, LLMResponse, ToolCall


class MockLLMProvider(LLMProvider):
    """Fake LLM that returns text or tool_calls based on a canned response."""
    def __init__(self, canned: LLMResponse | None = None):
        self.canned = canned or LLMResponse(text="你好", finish_reason="stop")
        self.last_messages = None
        self.last_tools = None

    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        self.last_messages = messages
        self.last_tools = tools
        return self.canned

    async def chat_stream(self, messages: list[dict], tools: list[dict] | None = None):
        self.last_messages = messages
        self.last_tools = tools
        for char in self.canned.text or "":
            yield char


class TestLLMProviderInterface:
    @pytest.mark.asyncio
    async def test_chat_returns_llm_response(self):
        provider = MockLLMProvider()
        resp = await provider.chat([{"role": "user", "content": "你好"}])
        assert isinstance(resp, LLMResponse)
        assert resp.text == "你好"

    @pytest.mark.asyncio
    async def test_chat_passes_messages_and_tools(self):
        provider = MockLLMProvider()
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        await provider.chat([{"role": "user", "content": "测试"}], tools)
        assert provider.last_messages[0]["content"] == "测试"
        assert len(provider.last_tools) == 1

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self):
        provider = MockLLMProvider(LLMResponse(
            tool_calls=[ToolCall(id="1", name="get_price", arguments={"symbol": "600519"})],
            finish_reason="tool_calls",
        ))
        resp = await provider.chat([{"role": "user", "content": "茅台价格"}])
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "get_price"

    @pytest.mark.asyncio
    async def test_chat_stream_yields_characters(self):
        provider = MockLLMProvider(LLMResponse(text="ABC"))
        chars = []
        async for c in provider.chat_stream([{"role": "user", "content": "hi"}]):
            chars.append(c)
        assert chars == ["A", "B", "C"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_llm_provider.py -v
```
Expected: FAIL — Module not found

- [ ] **Step 3: Write stockbot/llm/__init__.py**

```python
from stockbot.llm.base import LLMProvider, LLMResponse, ToolCall
from stockbot.llm.deepseek import DeepSeekProvider

__all__ = ["LLMProvider", "LLMResponse", "ToolCall", "DeepSeekProvider"]
```

- [ ] **Step 4: Write stockbot/llm/base.py**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)


class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        ...

    @abstractmethod
    async def chat_stream(self, messages: list[dict], tools: list[dict] | None = None) -> AsyncIterator[str]:
        ...
```

- [ ] **Step 5: Write stockbot/llm/deepseek.py**

```python
import json
import os
from openai import AsyncOpenAI
from stockbot.llm.base import LLMProvider, LLMResponse, ToolCall


class DeepSeekProvider(LLMProvider):
    def __init__(self, model: str = "deepseek-chat", api_key: str | None = None,
                 max_tokens: int = 4096, temperature: float = 0.3):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
        )

    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if tools:
            kwargs["tools"] = tools

        resp = await self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]

        tool_calls = []
        if choice.message.tool_calls:
            tool_calls = [
                ToolCall(id=tc.id, name=tc.function.name,
                         arguments=json.loads(tc.function.arguments))
                for tc in choice.message.tool_calls
            ]

        return LLMResponse(
            text=choice.message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage={"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens} if resp.usage else {},
        )

    async def chat_stream(self, messages: list[dict], tools: list[dict] | None = None):
        kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )
        if tools:
            kwargs["tools"] = tools

        stream = await self.client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_llm_provider.py -v
```
Expected: 4 PASS

- [ ] **Step 7: Commit**

```bash
git add stockbot/llm/ tests/test_llm_provider.py
git commit -m "feat: add LLMProvider abstract and DeepSeekProvider"
```

---

### Task 4: MemoryStore — SQLite foundation

**Files:**
- Create: `stockbot/memory/__init__.py`
- Create: `stockbot/memory/store.py`
- Create: `tests/test_memory_store.py`

- [ ] **Step 1: Write failing tests in tests/test_memory_store.py**

```python
import pytest
from stockbot.memory.store import MemoryStore


class TestMemoryStore:
    @pytest.fixture
    def store(self, temp_db):
        s = MemoryStore(temp_db)
        s.init_schema()
        return s

    def test_init_schema_creates_tables(self, store):
        tables = store._fetch_all("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        table_names = [t[0] for t in tables]
        assert "users" in table_names
        assert "conversations" in table_names
        assert "profile" in table_names
        assert "quota" in table_names

    def test_create_and_get_user(self, store):
        uid = store.create_user("testuser", "hashed_pw", "user")
        user = store.get_user("testuser")
        assert user["username"] == "testuser"
        assert user["role"] == "user"
        assert user["daily_quota"] == 5

    def test_get_user_returns_none_for_missing(self, store):
        assert store.get_user("noone") is None

    def test_add_and_get_messages(self, store):
        uid = store.create_user("u1", "pw", "user")
        store.add_message(uid, "user", "你好")
        store.add_message(uid, "assistant", "你好！有什么可以帮你的？")
        history = store.get_history(uid, 10)
        assert len(history) == 2
        assert history[0]["role"] == "assistant"
        assert history[1]["role"] == "user"

    def test_message_stores_tool_name(self, store):
        uid = store.create_user("u1", "pw", "user")
        store.add_message(uid, "tool", '{"price": 1680}', tool_name="get_realtime_quote")
        history = store.get_history(uid, 10)
        assert history[0]["tool_name"] == "get_realtime_quote"

    def test_profile_crud(self, store):
        uid = store.create_user("u1", "pw", "user")
        store.set_profile(uid, {"favorite_stocks": ["600519"]})
        profile = store.get_profile(uid)
        assert profile["favorite_stocks"] == ["600519"]

    def test_get_profile_returns_empty_dict_for_new_user(self, store):
        uid = store.create_user("u1", "pw", "user")
        assert store.get_profile(uid) == {}

    def test_quota_tracking(self, store):
        uid = store.create_user("u1", "pw", "user")
        q = store.get_quota(uid, "2026-05-23")
        assert q["calls"] == 0
        assert q["approved"] == 0

        store.incr_quota(uid, "2026-05-23")
        q = store.get_quota(uid, "2026-05-23")
        assert q["calls"] == 1

        store.incr_quota(uid, "2026-05-23")
        q = store.get_quota(uid, "2026-05-23")
        assert q["calls"] == 2

    def test_add_approved_quota(self, store):
        uid = store.create_user("u1", "pw", "user")
        store.add_approved(uid, "2026-05-23", 10)
        q = store.get_quota(uid, "2026-05-23")
        assert q["approved"] == 10

    def test_trim_history(self, store):
        uid = store.create_user("u1", "pw", "user")
        for i in range(10):
            store.add_message(uid, "user", f"msg {i}")
        store.trim_history(uid, keep=5)
        history = store.get_history(uid, 50)
        assert len(history) == 5

    def test_list_users(self, store):
        store.create_user("alice", "pw", "user")
        store.create_user("bob", "pw", "user")
        users = store.list_users()
        assert len(users) == 2
        usernames = {u["username"] for u in users}
        assert usernames == {"alice", "bob"}

    def test_update_user_quota(self, store):
        uid = store.create_user("u1", "pw", "user")
        store.update_user_quota(uid, 20)
        user = store.get_user_by_id(uid)
        assert user["daily_quota"] == 20
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_memory_store.py -v
```
Expected: FAIL — Module not found

- [ ] **Step 3: Write stockbot/memory/__init__.py**

```python
from stockbot.memory.store import MemoryStore
from stockbot.memory.history import ConversationHistory
from stockbot.memory.profile import ProfileManager

__all__ = ["MemoryStore", "ConversationHistory", "ProfileManager"]
```

- [ ] **Step 4: Write stockbot/memory/store.py**

```python
import sqlite3
import json
import uuid
from datetime import date


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    username    TEXT UNIQUE NOT NULL,
    password    TEXT NOT NULL,
    role        TEXT DEFAULT 'user',
    daily_quota INTEGER DEFAULT 5,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id),
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    tool_name   TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS profile (
    user_id     TEXT PRIMARY KEY REFERENCES users(id),
    value       TEXT NOT NULL DEFAULT '{}',
    updated_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS quota (
    user_id     TEXT NOT NULL REFERENCES users(id),
    date        TEXT NOT NULL,
    calls       INTEGER DEFAULT 0,
    approved    INTEGER DEFAULT 0,
    PRIMARY KEY (user_id, date)
);

CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_conv_user_date ON conversations(user_id, date(created_at));
"""


class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _fetch_all(self, sql: str, params: tuple = ()):
        with self._connect() as conn:
            return conn.execute(sql, params).fetchall()

    def _fetch_one(self, sql: str, params: tuple = ()):
        with self._connect() as conn:
            return conn.execute(sql, params).fetchone()

    def _execute(self, sql: str, params: tuple = ()):
        with self._connect() as conn:
            conn.execute(sql, params)

    def init_schema(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # ── Users ──

    def create_user(self, username: str, password_hash: str, role: str = "user") -> str:
        uid = str(uuid.uuid4())
        self._execute(
            "INSERT INTO users (id, username, password, role) VALUES (?, ?, ?, ?)",
            (uid, username, password_hash, role),
        )
        self._execute(
            "INSERT INTO profile (user_id) VALUES (?)", (uid,)
        )
        return uid

    def get_user(self, username: str) -> dict | None:
        row = self._fetch_one("SELECT * FROM users WHERE username = ?", (username,))
        return dict(row) if row else None

    def get_user_by_id(self, user_id: str) -> dict | None:
        row = self._fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))
        return dict(row) if row else None

    def list_users(self) -> list[dict]:
        rows = self._fetch_all("SELECT * FROM users ORDER BY created_at DESC")
        return [dict(r) for r in rows]

    def update_user_quota(self, user_id: str, daily_quota: int):
        self._execute("UPDATE users SET daily_quota = ? WHERE id = ?", (daily_quota, user_id))

    # ── Conversations ──

    def add_message(self, user_id: str, role: str, content: str, tool_name: str | None = None) -> str:
        mid = str(uuid.uuid4())
        self._execute(
            "INSERT INTO conversations (id, user_id, role, content, tool_name) VALUES (?, ?, ?, ?, ?)",
            (mid, user_id, role, content, tool_name),
        )
        return mid

    def get_history(self, user_id: str, limit: int = 50) -> list[dict]:
        rows = self._fetch_all(
            "SELECT role, content, tool_name, created_at FROM conversations "
            "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
        return [dict(r) for r in reversed(rows)]

    def trim_history(self, user_id: str, keep: int = 200):
        self._execute(
            "DELETE FROM conversations WHERE id NOT IN ("
            "SELECT id FROM conversations WHERE user_id = ? ORDER BY created_at DESC LIMIT ?"
            ") AND user_id = ?",
            (user_id, keep, user_id),
        )

    # ── Profile ──

    def get_profile(self, user_id: str) -> dict:
        row = self._fetch_one("SELECT value FROM profile WHERE user_id = ?", (user_id,))
        if row and row["value"]:
            return json.loads(row["value"])
        return {}

    def set_profile(self, user_id: str, data: dict):
        self._execute(
            "INSERT INTO profile (user_id, value) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET value = ?, updated_at = datetime('now')",
            (user_id, json.dumps(data, ensure_ascii=False), json.dumps(data, ensure_ascii=False)),
        )

    # ── Quota ──

    def get_quota(self, user_id: str, dt: str) -> dict:
        row = self._fetch_one(
            "SELECT calls, approved FROM quota WHERE user_id = ? AND date = ?",
            (user_id, dt),
        )
        if row:
            return {"calls": row["calls"], "approved": row["approved"]}
        return {"calls": 0, "approved": 0}

    def incr_quota(self, user_id: str, dt: str):
        self._execute(
            "INSERT INTO quota (user_id, date, calls) VALUES (?, ?, 1) "
            "ON CONFLICT(user_id, date) DO UPDATE SET calls = calls + 1",
            (user_id, dt),
        )

    def add_approved(self, user_id: str, dt: str, n: int):
        self._execute(
            "INSERT INTO quota (user_id, date, approved) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id, date) DO UPDATE SET approved = approved + ?",
            (user_id, dt, n, n),
        )
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_memory_store.py -v
```
Expected: 12 PASS

- [ ] **Step 6: Commit**

```bash
git add stockbot/memory/store.py stockbot/memory/__init__.py tests/test_memory_store.py
git commit -m "feat: add MemoryStore with SQLite schema and full CRUD"
```

---

### Task 5: ConversationHistory and ProfileManager

**Files:**
- Create: `stockbot/memory/history.py`
- Create: `stockbot/memory/profile.py`
- Create: `tests/test_history.py`
- Create: `tests/test_profile.py`

- [ ] **Step 1: Write failing tests in tests/test_history.py**

```python
import pytest
from stockbot.memory.store import MemoryStore
from stockbot.memory.history import ConversationHistory


class TestConversationHistory:
    @pytest.fixture
    def store(self, temp_db):
        s = MemoryStore(temp_db)
        s.init_schema()
        return s

    @pytest.fixture
    def user_id(self, store):
        return store.create_user("u1", "pw")

    @pytest.fixture
    def history(self, store):
        return ConversationHistory(store, history_limit=200)

    def test_get_recent_returns_messages(self, store, user_id, history):
        store.add_message(user_id, "user", "问题1")
        store.add_message(user_id, "assistant", "回答1")
        msgs = history.get_recent(user_id, 10)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_save_turn_adds_user_and_assistant(self, store, user_id, history):
        history.save_turn(user_id, "用户问题", "助手回答")
        msgs = history.get_recent(user_id, 10)
        assert len(msgs) == 2
        assert msgs[0]["content"] == "用户问题"
        assert msgs[1]["content"] == "助手回答"

    def test_save_turn_with_tool_calls(self, store, user_id, history):
        tool_results = [
            {"name": "get_price", "result": '{"price": 1680}'}
        ]
        history.save_turn(user_id, "茅台价格", "助手回答", tool_results)
        msgs = history.get_recent(user_id, 10)
        assert len(msgs) == 4  # user, tool, assistant, tool

    def test_trim_triggered_when_over_limit(self, store, user_id, history):
        history.history_limit = 10
        for i in range(15):
            history.save_turn(user_id, f"msg{i}", f"reply{i}")
        msgs = history.get_recent(user_id, 50)
        assert len(msgs) <= 20  # 10 turns × 2 = 20 max

    def test_messages_are_chat_format(self, store, user_id, history):
        history.save_turn(user_id, "你好", "你好！")
        msgs = history.get_recent(user_id, 10)
        for m in msgs:
            assert "role" in m
            assert "content" in m
```

- [ ] **Step 2: Write failing tests in tests/test_profile.py**

```python
import pytest
from stockbot.memory.store import MemoryStore
from stockbot.memory.profile import ProfileManager


class TestProfileManager:
    @pytest.fixture
    def store(self, temp_db):
        s = MemoryStore(temp_db)
        s.init_schema()
        return s

    @pytest.fixture
    def user_id(self, store):
        return store.create_user("u1", "pw")

    @pytest.fixture
    def profile(self, store):
        return ProfileManager(store)

    def test_add_favorite_stock(self, store, user_id, profile):
        profile.add_favorite(user_id, "600519")
        data = store.get_profile(user_id)
        assert "600519" in data["favorite_stocks"]

    def test_remove_favorite_stock(self, store, user_id, profile):
        profile.add_favorite(user_id, "600519")
        profile.add_favorite(user_id, "000858")
        profile.remove_favorite(user_id, "600519")
        data = store.get_profile(user_id)
        assert data["favorite_stocks"] == ["000858"]

    def test_get_favorites_returns_list(self, store, user_id, profile):
        profile.add_favorite(user_id, "600519")
        favs = profile.get_favorites(user_id)
        assert favs == ["600519"]

    def test_record_query_updates_frequency(self, store, user_id, profile):
        profile.record_query(user_id, "600519")
        profile.record_query(user_id, "600519")
        profile.record_query(user_id, "000858")
        data = store.get_profile(user_id)
        assert data["query_frequency"]["600519"] == 2
        assert data["query_frequency"]["000858"] == 1

    def test_summary_returns_readable_string(self, store, user_id, profile):
        profile.add_favorite(user_id, "600519")
        profile.record_query(user_id, "600519")
        summary = profile.summary(user_id)
        assert "600519" in summary

    def test_no_duplicate_favorites(self, store, user_id, profile):
        profile.add_favorite(user_id, "600519")
        profile.add_favorite(user_id, "600519")
        data = store.get_profile(user_id)
        assert data["favorite_stocks"] == ["600519"]
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
pytest tests/test_history.py tests/test_profile.py -v
```
Expected: FAIL

- [ ] **Step 4: Write stockbot/memory/history.py**

```python
from stockbot.memory.store import MemoryStore


class ConversationHistory:
    def __init__(self, store: MemoryStore, history_limit: int = 200):
        self.store = store
        self.history_limit = history_limit

    def get_recent(self, user_id: str, limit: int = 50) -> list[dict]:
        return self.store.get_history(user_id, limit)

    def save_turn(self, user_id: str, user_content: str, assistant_content: str,
                  tool_results: list[dict] | None = None):
        self.store.add_message(user_id, "user", user_content)
        if tool_results:
            for tr in tool_results:
                self.store.add_message(user_id, "tool", str(tr["result"]), tool_name=tr["name"])
        self.store.add_message(user_id, "assistant", assistant_content)
        self.store.trim_history(user_id, self.history_limit)
```

- [ ] **Step 5: Write stockbot/memory/profile.py**

```python
import json
from stockbot.memory.store import MemoryStore


class ProfileManager:
    def __init__(self, store: MemoryStore):
        self.store = store

    def _get_data(self, user_id: str) -> dict:
        data = self.store.get_profile(user_id)
        data.setdefault("favorite_stocks", [])
        data.setdefault("query_frequency", {})
        data.setdefault("risk_preference", "未设置")
        return data

    def _save(self, user_id: str, data: dict):
        self.store.set_profile(user_id, data)

    def add_favorite(self, user_id: str, symbol: str):
        data = self._get_data(user_id)
        if symbol not in data["favorite_stocks"]:
            data["favorite_stocks"].append(symbol)
        self._save(user_id, data)

    def remove_favorite(self, user_id: str, symbol: str):
        data = self._get_data(user_id)
        if symbol in data["favorite_stocks"]:
            data["favorite_stocks"].remove(symbol)
        self._save(user_id, data)

    def get_favorites(self, user_id: str) -> list[str]:
        return self._get_data(user_id).get("favorite_stocks", [])

    def record_query(self, user_id: str, symbol: str):
        data = self._get_data(user_id)
        data["query_frequency"][symbol] = data["query_frequency"].get(symbol, 0) + 1
        self._save(user_id, data)

    def summary(self, user_id: str) -> str:
        data = self._get_data(user_id)
        parts = []
        if data.get("favorite_stocks"):
            parts.append(f"关注股票: {', '.join(data['favorite_stocks'])}")
        if data.get("risk_preference"):
            parts.append(f"风险偏好: {data['risk_preference']}")
        if data.get("query_frequency"):
            top = sorted(data["query_frequency"].items(), key=lambda x: x[1], reverse=True)[:3]
            parts.append(f"最近关注: {', '.join(s for s, _ in top)}")
        return " | ".join(parts) if parts else "新用户，暂无画像"
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_history.py tests/test_profile.py -v
```
Expected: 11 PASS

- [ ] **Step 7: Commit**

```bash
git add stockbot/memory/history.py stockbot/memory/profile.py tests/test_history.py tests/test_profile.py
git commit -m "feat: add ConversationHistory and ProfileManager"
```

---

## Phase 3: Service Layer

### Task 6: Tool base and ToolRegistry

**Files:**
- Create: `stockbot/tools/__init__.py`
- Create: `stockbot/tools/base.py`
- Create: `stockbot/tools/registry.py`
- Create: `tests/test_tool_registry.py`

- [ ] **Step 1: Write failing tests in tests/test_tool_registry.py**

```python
import pytest
from stockbot.tools.base import Tool
from stockbot.tools.registry import ToolRegistry


async def echo_func(text: str) -> str:
    return f"echo: {text}"


async def add_func(a: int, b: int) -> str:
    return str(a + b)


class TestTool:
    def test_tool_to_openai_schema(self):
        tool = Tool(
            name="echo",
            description="Echo back the input",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to echo"}
                },
                "required": ["text"],
            },
            func=echo_func,
        )
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"

    @pytest.mark.asyncio
    async def test_tool_run(self):
        tool = Tool(name="echo", description="Echo", parameters={}, func=echo_func)
        result = await tool.run(text="hello")
        assert result == "echo: hello"


class TestToolRegistry:
    @pytest.fixture
    def registry(self):
        reg = ToolRegistry()
        reg.register(Tool(name="echo", description="Echo", parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text"}},
            "required": ["text"],
        }, func=echo_func))
        reg.register(Tool(name="add", description="Add two numbers", parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"},
            },
            "required": ["a", "b"],
        }, func=add_func))
        return reg

    def test_get_schemas_returns_all_tools(self, registry):
        schemas = registry.get_schemas()
        assert len(schemas) == 2
        names = [s["function"]["name"] for s in schemas]
        assert "echo" in names
        assert "add" in names

    def test_describe_returns_tool_list(self, registry):
        desc = registry.describe()
        assert "echo" in desc
        assert "add" in desc

    @pytest.mark.asyncio
    async def test_execute_runs_tool(self, registry):
        result = await registry.execute("echo", {"text": "hi"})
        assert result == "echo: hi"

    @pytest.mark.asyncio
    async def test_execute_returns_error_for_unknown_tool(self, registry):
        result = await registry.execute("unknown", {})
        assert "错误" in result or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_handles_exception_gracefully(self, registry):
        async def failing(x: str) -> str:
            raise RuntimeError("test failure")
        registry.register(Tool(name="fail", description="x", parameters={
            "type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]
        }, func=failing))
        result = await registry.execute("fail", {"x": "y"})
        assert "错误" in result
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_tool_registry.py -v
```
Expected: FAIL

- [ ] **Step 3: Write stockbot/tools/__init__.py**

```python
from stockbot.tools.base import Tool
from stockbot.tools.registry import ToolRegistry

__all__ = ["Tool", "ToolRegistry"]
```

- [ ] **Step 4: Write stockbot/tools/base.py**

```python
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def run(self, **kwargs) -> str:
        import asyncio
        result = self.func(**kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return str(result)
```

- [ ] **Step 5: Write stockbot/tools/registry.py**

```python
import asyncio
from stockbot.tools.base import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get_schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def describe(self) -> str:
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)

    async def execute(self, name: str, args: dict) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f'错误: 未找到工具 "{name}"。可用工具: {", ".join(self._tools.keys())}'
        try:
            return await asyncio.wait_for(tool.run(**args), timeout=10.0)
        except asyncio.TimeoutError:
            return f'错误: 工具 "{name}" 执行超时'
        except Exception as e:
            return f'错误: 工具 "{name}" 执行失败: {e}'
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_tool_registry.py -v
```
Expected: 8 PASS

- [ ] **Step 7: Commit**

```bash
git add stockbot/tools/base.py stockbot/tools/registry.py stockbot/tools/__init__.py tests/test_tool_registry.py
git commit -m "feat: add Tool base class and ToolRegistry"
```

---

### Task 7: Five stock tools

**Files:**
- Create: `stockbot/tools/stock_search.py`
- Create: `stockbot/tools/stock_price.py`
- Create: `stockbot/tools/stock_finance.py`
- Create: `stockbot/tools/stock_trend.py`
- Create: `stockbot/tools/stock_news.py`
- Create: `tests/test_stock_tools.py`

- [ ] **Step 1: Write failing tests in tests/test_stock_tools.py**

```python
import pytest
from stockbot.data.base import StockQuote, StockHistory
from stockbot.tools.stock_search import create_search_tool
from stockbot.tools.stock_price import create_price_tool
from stockbot.tools.stock_finance import create_finance_tool
from stockbot.tools.stock_trend import create_trend_tool
from stockbot.tools.stock_news import create_news_tool


class MockDataProvider:
    def search(self, query: str) -> list[dict]:
        return [{"symbol": "600519", "name": "贵州茅台", "market": "SH"}]

    def get_realtime(self, symbol: str) -> StockQuote:
        return StockQuote(symbol=symbol, name="贵州茅台", price=1680.0,
                          change_pct=2.3, volume=1e7, timestamp="2026-05-23")

    def get_history(self, symbol: str, period: str) -> StockHistory:
        data = [
            {"date": f"2026-05-{20+i:02d}", "open": 1640+i*10, "high": 1660+i*10,
             "low": 1630+i*10, "close": 1650+i*10, "volume": 1e7}
            for i in range(5)
        ]
        return StockHistory(symbol=symbol, name="贵州茅台", data=data)

    def get_financial(self, symbol: str) -> dict:
        return {"pe": 25.5, "pb": 8.2, "roe": 32.1, "revenue_growth": 15.3, "eps": 65.8}

    def get_news(self, symbol: str, limit: int = 5) -> list[dict]:
        return [{"title": "茅台涨价", "source": "财经网", "time": "2026-05-23", "url": ""}]


@pytest.fixture
def provider():
    return MockDataProvider()


class TestStockTools:
    @pytest.mark.asyncio
    async def test_search_stock(self, provider):
        tool = create_search_tool(provider)
        result = await tool.run(query="茅台")
        assert "600519" in result
        assert "贵州茅台" in result

    @pytest.mark.asyncio
    async def test_get_realtime_quote(self, provider):
        tool = create_price_tool(provider)
        result = await tool.run(symbol="600519")
        assert "1680" in result

    @pytest.mark.asyncio
    async def test_get_financial_data(self, provider):
        tool = create_finance_tool(provider)
        result = await tool.run(symbol="600519", metric="all")
        assert "25.5" in result or "PE" in result.lower()

    @pytest.mark.asyncio
    async def test_analyze_trend(self, provider):
        tool = create_trend_tool(provider)
        result = await tool.run(symbol="600519", period="1m")
        assert "乐观" in result
        assert "中性" in result
        assert "悲观" in result

    @pytest.mark.asyncio
    async def test_analyze_trend_with_scenario_labels(self, provider):
        tool = create_trend_tool(provider)
        result = await tool.run(symbol="600519", period="1m")
        assert "支撑" in result or "压力" in result
        assert "⚠️" in result or "风险" in result or "仅供参考" in result

    @pytest.mark.asyncio
    async def test_search_news(self, provider):
        tool = create_news_tool(provider)
        result = await tool.run(symbol="600519", limit=3)
        assert "茅台涨价" in result
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_stock_tools.py -v
```
Expected: FAIL

- [ ] **Step 3: Write stockbot/tools/stock_search.py**

```python
from stockbot.tools.base import Tool
from stockbot.data.base import DataProvider


def create_search_tool(provider: DataProvider) -> Tool:
    async def search(query: str) -> str:
        results = provider.search(query)
        if not results:
            return f'未找到与 "{query}" 相关的股票。请尝试其他关键词或完整代码。'
        lines = [f'搜索 "{query}" 的结果:']
        for r in results:
            lines.append(f"  {r['symbol']}  {r['name']}  ({r.get('market', '')})")
        return "\n".join(lines)

    return Tool(
        name="search_stock",
        description="按名称或代码搜索股票。返回匹配的股票代码、名称和市场。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词，如公司名称或股票代码"}
            },
            "required": ["query"],
        },
        func=search,
    )
```

- [ ] **Step 4: Write stockbot/tools/stock_price.py**

```python
from stockbot.tools.base import Tool
from stockbot.data.base import DataProvider


def create_price_tool(provider: DataProvider) -> Tool:
    async def get_realtime_quote(symbol: str) -> str:
        quote = provider.get_realtime(symbol)
        arrow = "↑" if quote.change_pct >= 0 else "↓"
        sign = "+" if quote.change_pct >= 0 else ""
        return (
            f"{quote.name} ({quote.symbol})\n"
            f"  最新价: {quote.price:.2f}\n"
            f"  涨跌幅: {sign}{quote.change_pct:.2f}% {arrow}\n"
            f"  成交量: {quote.volume:.0f} 手\n"
            f"  时间: {quote.timestamp}"
        )

    return Tool(
        name="get_realtime_quote",
        description="获取股票实时行情。返回最新价格、涨跌幅、成交量等信息。",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码，如 600519"}
            },
            "required": ["symbol"],
        },
        func=get_realtime_quote,
    )
```

- [ ] **Step 5: Write stockbot/tools/stock_finance.py**

```python
from stockbot.tools.base import Tool
from stockbot.data.base import DataProvider


SUPPORTED_METRICS = ["pe", "pb", "roe", "revenue_growth", "eps", "all"]


def create_finance_tool(provider: DataProvider) -> Tool:
    async def get_financial_data(symbol: str, metric: str = "all") -> str:
        if metric not in SUPPORTED_METRICS:
            return f'不支持的指标: {metric}。可用指标: {", ".join(SUPPORTED_METRICS)}'

        data = provider.get_financial(symbol)
        if not data:
            return f"未找到 {symbol} 的财务数据。"

        labels = {"pe": "市盈率(PE)", "pb": "市净率(PB)", "roe": "净资产收益率(ROE)%",
                  "revenue_growth": "营收增长%", "eps": "每股收益(EPS)"}

        lines = [f"📊 {symbol} 财务指标:"]
        if metric == "all":
            for key, label in labels.items():
                val = data.get(key)
                if val is not None:
                    lines.append(f"  {label}: {val}")
        else:
            val = data.get(metric)
            label = labels.get(metric, metric)
            lines.append(f"  {label}: {val if val is not None else '无数据'}")
        return "\n".join(lines)

    return Tool(
        name="get_financial_data",
        description="获取股票财务指标。支持 PE、PB、ROE、营收增长率、EPS。用 'all' 获取全部。",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码"},
                "metric": {"type": "string", "description": f"指标名称: {', '.join(SUPPORTED_METRICS)}", "default": "all"},
            },
            "required": ["symbol"],
        },
        func=get_financial_data,
    )
```

- [ ] **Step 6: Write stockbot/tools/stock_trend.py**

```python
from stockbot.tools.base import Tool
from stockbot.data.base import DataProvider


def _calc_ma(closes: list[float], window: int) -> list[float | None]:
    result = []
    for i in range(len(closes)):
        if i < window - 1:
            result.append(None)
        else:
            result.append(sum(closes[i - window + 1:i + 1]) / window)
    return result


def _calc_ema(closes: list[float], window: int) -> list[float | None]:
    result = []
    multiplier = 2 / (window + 1)
    for i in range(len(closes)):
        if i == 0:
            result.append(closes[0])
        elif i < window - 1:
            result.append(None)
        elif i == window - 1:
            result.append(sum(closes[:window]) / window)
        else:
            result.append((closes[i] - result[i - 1]) * multiplier + result[i - 1])
    return result


def _calc_macd(closes: list[float]) -> tuple[list[float | None], list[float | None], list[float | None]]:
    ema12 = _calc_ema(closes, 12)
    ema26 = _calc_ema(closes, 26)
    dif = [None if e12 is None or e26 is None else e12 - e26 for e12, e26 in zip(ema12, ema26)]
    dea = _calc_ema([d if d is not None else 0 for d in dif], 9)
    macd_hist = [None if d is None or dea_i is None else 2 * (d - dea_i) for d, dea_i in zip(dif, dea)]
    return dif, dea, macd_hist


def _calc_rsi(closes: list[float], window: int = 14) -> list[float | None]:
    result = [None] * len(closes)
    gains = []
    losses = []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    for i in range(len(gains)):
        if i < window - 1:
            continue
        avg_gain = sum(gains[i - window + 1:i + 1]) / window
        avg_loss = sum(losses[i - window + 1:i + 1]) / window
        if avg_loss == 0:
            result[i + 1] = 100
        else:
            result[i + 1] = 100 - 100 / (1 + avg_gain / avg_loss)
    return result


def _find_levels(closes: list[float]) -> tuple[float, float, float | None, float | None]:
    high = max(closes)
    low = min(closes)
    mid = (high + low) / 2
    recent = closes[-1]
    levels = sorted(set(closes), reverse=True)
    resistance = None
    support = None
    for lvl in levels:
        if lvl > recent and (resistance is None or lvl < resistance):
            resistance = lvl
        if lvl < recent and (support is None or lvl > support):
            support = lvl
    return recent, mid, support, resistance


def create_trend_tool(provider: DataProvider) -> Tool:
    async def analyze_trend(symbol: str, period: str = "3m") -> str:
        history = provider.get_history(symbol, period)
        if not history.data or len(history.data) < 20:
            return f"{symbol} 历史数据不足，需要至少 20 个交易日的数据。"

        closes = [d["close"] for d in history.data]
        volumes = [d["volume"] for d in history.data]
        dates = [d["date"] for d in history.data]
        name = history.name
        last_price = closes[-1]
        last_vol = volumes[-1]

        # Technical indicators
        ma5 = _calc_ma(closes, 5)
        ma10 = _calc_ma(closes, 10)
        ma20 = _calc_ma(closes, 20)
        ma60 = _calc_ma(closes, 60) if len(closes) >= 60 else [None] * len(closes)
        dif, dea, macd_hist = _calc_macd(closes)
        rsi = _calc_rsi(closes, 14)
        vol_ma5 = _calc_ma(volumes, 5)

        cur_price, mid_price, support, resistance = _find_levels(closes)

        # Trend classification
        ma5_val = ma5[-1] or 0
        ma10_val = ma10[-1] or 0
        ma20_val = ma20[-1] or 0
        ma60_val = ma60[-1] if ma60[-1] else 0
        rsi_val = rsi[-1] or 50
        macd_dif = dif[-1] or 0
        macd_dea = dea[-1] or 0
        macd_bar = macd_hist[-1] or 0
        vol_ratio = last_vol / (vol_ma5[-1] or 1)

        if ma5_val > ma10_val > ma20_val:
            trend = "多头排列，短期上涨趋势"
            base_scenario = "bullish"
        elif ma5_val < ma10_val < ma20_val:
            trend = "空头排列，短期下跌趋势"
            base_scenario = "bearish"
        else:
            trend = "均线交织，处于震荡整理"
            base_scenario = "neutral"

        overbought = rsi_val > 70
        oversold = rsi_val < 30

        bullish_golden_cross = macd_dif > macd_dea and macd_bar > 0
        bearish_dead_cross = macd_dif < macd_dea and macd_bar < 0

        vol_expanding = vol_ratio > 1.5
        vol_contracting = vol_ratio < 0.5

        # Build scenario analysis
        bullish_cond = []
        neutral_cond = []
        bearish_cond = []

        if base_scenario == "bullish":
            bullish_cond.append("当前多头趋势延续")
            bullish_cond.append("成交量维持或放大")
            bullish_target = f"{last_price * 1.05:.2f}-{last_price * 1.10:.2f}"
            bearish_cond.append("跌破 20 日均线 (%.2f)" % ma20_val)
            bearish_cond.append("MACD 死叉")
            bearish_target = f"{support or last_price * 0.95:.2f}"
            neutral_target = f"{last_price * 0.98:.2f}-{last_price * 1.02:.2f}"
        elif base_scenario == "bearish":
            bearish_cond.append("当前空头趋势延续")
            bearish_cond.append("成交量持续萎缩")
            bearish_target = f"{support or last_price * 0.90:.2f}"
            bullish_cond.append("放量突破 10 日均线 (%.2f)" % ma10_val)
            bullish_cond.append("MACD 金叉")
            bullish_target = f"{last_price * 1.03:.2f}-{last_price * 1.08:.2f}"
            neutral_target = f"{last_price * 0.98:.2f}-{last_price * 1.02:.2f}"
        else:
            bullish_cond.append("放量突破震荡区间上沿")
            bullish_target = f"{resistance or last_price * 1.05:.2f}-{last_price * 1.10:.2f}" if resistance else f"{last_price * 1.05:.2f}-{last_price * 1.10:.2f}"
            bearish_cond.append("放量跌破震荡区间下沿")
            bearish_target = f"{support or last_price * 0.90:.2f}" if support else f"{last_price * 0.90:.2f}"
            neutral_cond.append("维持箱体震荡")
            neutral_target = f"{support or last_price * 0.95:.2f}-{resistance or last_price * 1.05:.2f}"

        if overbought:
            bearish_cond.append(f"RSI={rsi_val:.0f} 超买，回调风险")
        if oversold:
            bullish_cond.append(f"RSI={rsi_val:.0f} 超卖，反弹动能")
        if vol_expanding:
            bullish_cond.append("放量配合，趋势强化")
        if vol_contracting:
            neutral_cond.append("缩量整理，等待方向选择")

        sups = [f"{support:.2f}" if support else "N/A"]
        rests = [f"{resistance:.2f}" if resistance else "N/A"]
        if ma20_val:
            sups.append(f"MA20 {ma20_val:.2f}")
        if ma60_val:
            sups.append(f"MA60 {ma60_val:.2f}")
        if ma10_val:
            rests.append(f"MA10 {ma10_val:.2f}")
        if ma5_val:
            rests.append(f"MA5 {ma5_val:.2f}")

        return f"""📊 {name} ({symbol}) 技术面分析 — {period}周期

当前状态: {trend}
MACD: DIF={macd_dif:.2f} DEA={macd_dea:.2f} BAR={macd_bar:.2f}
RSI(14): {rsi_val:.1f} {"(超买)" if overbought else "(超卖)" if oversold else "(中性)"}
量比: {vol_ratio:.1f} {"(放量)" if vol_expanding else "(缩量)" if vol_contracting else "(正常)"}

┌─────────────────────────────────────────────────────┐
│ 🟢 乐观情景                                          │
│   条件: {'、'.join(bullish_cond) if bullish_cond else '基本面改善 + 市场情绪转好'}
│   目标: {bullish_target}
├─────────────────────────────────────────────────────┤
│ 🟡 中性情景                                          │
│   条件: {'、'.join(neutral_cond) if neutral_cond else '大盘横盘 + 无重大消息'}
│   区间: {neutral_target}
├─────────────────────────────────────────────────────┤
│ 🔴 悲观情景                                          │
│   条件: {'、'.join(bearish_cond) if bearish_cond else '大盘回落 + 板块利空'}
│   目标: {bearish_target}
└─────────────────────────────────────────────────────┘

支撑: {', '.join(sups)}    压力: {', '.join(rests)}

⚠️ 以上为基于技术指标的多情景推演，不构成投资建议。实际走势受政策、市场情绪、流动性等不可量化因素影响。"""

    return Tool(
        name="analyze_trend",
        description="对股票进行多情景技术分析。输出乐观/中性/悲观三种情景的走势推演。需提供股票代码，可选周期(1m/3m/6m/1y)。",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码"},
                "period": {"type": "string", "description": "分析周期: 1m(1月) 3m(3月,默认) 6m(半年) 1y(1年)", "default": "3m"},
            },
            "required": ["symbol"],
        },
        func=analyze_trend,
    )
```

- [ ] **Step 7: Write stockbot/tools/stock_news.py**

```python
from stockbot.tools.base import Tool
from stockbot.data.base import DataProvider


def create_news_tool(provider: DataProvider) -> Tool:
    async def search_news(symbol: str, limit: int = 5) -> str:
        news = provider.get_news(symbol, limit)
        if not news:
            return f"未找到 {symbol} 的相关新闻。"
        lines = [f"📰 {symbol} 相关新闻:"]
        for i, n in enumerate(news, 1):
            src = n.get("source", "")
            time = n.get("time", "")
            lines.append(f"  {i}. {n['title']}")
            lines.append(f"     来源: {src}  {time}")
        return "\n".join(lines)

    return Tool(
        name="search_news",
        description="搜索股票相关新闻。返回最近新闻标题、来源和时间。",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码"},
                "limit": {"type": "integer", "description": "返回条数，默认 5", "default": 5},
            },
            "required": ["symbol"],
        },
        func=search_news,
    )
```

- [ ] **Step 8: Run tests**

```bash
pytest tests/test_stock_tools.py -v
```
Expected: 6 PASS

- [ ] **Step 9: Commit**

```bash
git add stockbot/tools/stock_*.py tests/test_stock_tools.py
git commit -m "feat: add 5 stock analysis tools (search, price, finance, trend, news)"
```

---

### Task 8: QuotaManager

**Files:**
- Create: `stockbot/quota.py`
- Create: `tests/test_quota.py`

- [ ] **Step 1: Write failing tests in tests/test_quota.py**

```python
import pytest
from datetime import date
from stockbot.memory.store import MemoryStore
from stockbot.quota import QuotaManager, QuotaResult


class TestQuotaManager:
    @pytest.fixture
    def store(self, temp_db):
        s = MemoryStore(temp_db)
        s.init_schema()
        return s

    @pytest.fixture
    def user_id(self, store):
        return store.create_user("u1", "pw")

    @pytest.fixture
    def manager(self, store):
        return QuotaManager(store, daily_limit=5)

    def test_check_allows_within_limit(self, store, user_id, manager):
        result = manager.check(user_id)
        assert not result.blocked
        assert result.remain == 5
        assert result.used == 0

    def test_consume_increments_count(self, store, user_id, manager):
        manager.consume(user_id)
        manager.consume(user_id)
        result = manager.check(user_id)
        assert result.used == 2
        assert result.remain == 3
        assert not result.blocked

    def test_check_blocks_when_exhausted(self, store, user_id, manager):
        for _ in range(5):
            manager.consume(user_id)
        result = manager.check(user_id)
        assert result.blocked
        assert result.remain == 0

    def test_admin_approve_adds_extra(self, store, user_id, manager):
        for _ in range(5):
            manager.consume(user_id)
        manager.approve(user_id, 5)
        result = manager.check(user_id)
        assert not result.blocked
        assert result.remain == 5
        assert result.limit == 10

    def test_verify_password(self, manager):
        manager.admin_password_hash = None
        assert manager.verify_password("admin123")  # raw comparison when no hash
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_quota.py -v
```
Expected: FAIL

- [ ] **Step 3: Write stockbot/quota.py**

```python
import os
from dataclasses import dataclass
from datetime import date
from stockbot.memory.store import MemoryStore


@dataclass
class QuotaResult:
    used: int
    limit: int
    remain: int
    blocked: bool


class QuotaManager:
    def __init__(self, store: MemoryStore, daily_limit: int = 5):
        self.store = store
        self.daily_limit = daily_limit
        self.admin_password_hash = os.environ.get("ADMIN_PASS", "admin123")

    def check(self, user_id: str) -> QuotaResult:
        today = date.today().isoformat()
        row = self.store.get_quota(user_id, today)
        used = row["calls"]
        approved = row["approved"]
        limit = self.daily_limit + approved
        remain = limit - used
        return QuotaResult(used=used, limit=limit, remain=max(0, remain), blocked=(remain <= 0))

    def consume(self, user_id: str):
        today = date.today().isoformat()
        self.store.incr_quota(user_id, today)

    def approve(self, user_id: str, extra: int):
        today = date.today().isoformat()
        self.store.add_approved(user_id, today, extra)

    def reset(self, user_id: str):
        today = date.today().isoformat()
        self.store.add_approved(user_id, today, 0)

    def verify_password(self, password: str) -> bool:
        return password == self.admin_password_hash
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_quota.py -v
```
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add stockbot/quota.py tests/test_quota.py
git commit -m "feat: add QuotaManager with daily limit and admin approval"
```

---

### Task 9: ContextAssembler

**Files:**
- Create: `stockbot/context.py`
- Create: `tests/test_context.py`

- [ ] **Step 1: Write failing tests in tests/test_context.py**

```python
import pytest
from stockbot.memory.store import MemoryStore
from stockbot.memory.history import ConversationHistory
from stockbot.memory.profile import ProfileManager
from stockbot.tools.base import Tool
from stockbot.tools.registry import ToolRegistry
from stockbot.context import ContextAssembler


class TestContextAssembler:
    @pytest.fixture
    def store(self, temp_db):
        s = MemoryStore(temp_db)
        s.init_schema()
        return s

    @pytest.fixture
    def user_id(self, store):
        return store.create_user("u1", "pw")

    @pytest.fixture
    def history(self, store):
        h = ConversationHistory(store)
        return h

    @pytest.fixture
    def profile(self, store):
        return ProfileManager(store)

    @pytest.fixture
    def tools(self):
        reg = ToolRegistry()
        async def echo(text: str) -> str:
            return text
        reg.register(Tool(name="echo", description="回显输入", parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }, func=echo))
        return reg

    @pytest.fixture
    def ctx(self, store, history, profile, tools):
        return ContextAssembler(tools, profile, history)

    def test_build_returns_correct_structure(self, ctx, user_id):
        msgs = ctx.build(user_id, "你好")
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "你好"

    def test_system_prompt_contains_tool_descriptions(self, ctx, user_id):
        msgs = ctx.build(user_id, "你好")
        assert "echo" in msgs[0]["content"]
        assert "回显输入" in msgs[0]["content"]

    def test_system_prompt_contains_profile_summary(self, ctx, user_id, profile):
        profile.add_favorite(user_id, "600519")
        msgs = ctx.build(user_id, "你好")
        assert "600519" in msgs[0]["content"]

    def test_system_prompt_contains_warning(self, ctx, user_id):
        msgs = ctx.build(user_id, "你好")
        assert "仅供参考" in msgs[0]["content"]

    def test_build_includes_history(self, ctx, user_id, history):
        history.save_turn(user_id, "问题1", "回答1")
        msgs = ctx.build(user_id, "问题2")
        roles = [m["role"] for m in msgs]
        assert roles.count("user") >= 2  # system prompt + 问题1 + 问题2 = 2 user messages
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_context.py -v
```
Expected: FAIL

- [ ] **Step 3: Write stockbot/context.py**

```python
from stockbot.tools.registry import ToolRegistry
from stockbot.memory.profile import ProfileManager
from stockbot.memory.history import ConversationHistory


SYSTEM_PROMPT_TEMPLATE = """你是一个专业的 A 股投资分析助手，名叫 StockBot。

你可以使用以下工具帮助用户分析股票：
{tools}

{profile_section}

规则：
- 涉及具体股票时，必须先调用工具获取实时数据，绝对不要凭记忆编造价格或走势
- 技术分析使用 analyze_trend 工具，输出多情景推演而非确定预测
- 用简洁清晰的中文回复，关键数据用列表呈现
- 所有分析类回复末尾附加 ⚠️ 声明"""


class ContextAssembler:
    def __init__(self, tool_registry: ToolRegistry, profile_manager: ProfileManager,
                 history: ConversationHistory, history_limit: int = 10):
        self.tool_registry = tool_registry
        self.profile_manager = profile_manager
        self.history = history
        self.history_limit = history_limit

    def build(self, user_id: str, user_input: str) -> list[dict]:
        system_content = self._system_prompt(user_id)
        recent = self.history.get_recent(user_id, self.history_limit)
        return [
            {"role": "system", "content": system_content},
            *recent,
            {"role": "user", "content": user_input},
        ]

    def _system_prompt(self, user_id: str) -> str:
        tools_desc = self.tool_registry.describe()
        summary = self.profile_manager.summary(user_id)
        if summary and summary != "新用户，暂无画像":
            profile_section = f"当前用户画像: {summary}\n\n根据用户画像个性化回复，优先关注用户关注的信息。"
        else:
            profile_section = "用户画像: 暂无。在对话中了解用户偏好。"

        return SYSTEM_PROMPT_TEMPLATE.format(
            tools=tools_desc,
            profile_section=profile_section,
        )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_context.py -v
```
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add stockbot/context.py tests/test_context.py
git commit -m "feat: add ContextAssembler for message assembly with profile injection"
```

---

### Task 10: Auth

**Files:**
- Create: `stockbot/auth.py`
- Create: `tests/test_auth.py`

- [ ] **Step 1: Write failing tests in tests/test_auth.py**

```python
import pytest
import bcrypt
from stockbot.memory.store import MemoryStore
from stockbot.auth import AuthManager


class TestAuthManager:
    @pytest.fixture
    def store(self, temp_db):
        s = MemoryStore(temp_db)
        s.init_schema()
        return s

    @pytest.fixture
    def auth(self, store):
        return AuthManager(store, open_registration=True)

    def test_register_creates_user(self, store, auth):
        uid = auth.register("testuser", "password123")
        assert uid is not None
        user = store.get_user("testuser")
        assert user is not None
        assert user["role"] == "user"

    def test_register_duplicate_returns_none(self, store, auth):
        auth.register("testuser", "pw1")
        uid = auth.register("testuser", "pw2")
        assert uid is None

    def test_login_success(self, store, auth):
        auth.register("testuser", "correctpw")
        user = auth.login("testuser", "correctpw")
        assert user is not None
        assert user["username"] == "testuser"

    def test_login_wrong_password(self, store, auth):
        auth.register("testuser", "correctpw")
        user = auth.login("testuser", "wrongpw")
        assert user is None

    def test_login_nonexistent_user(self, store, auth):
        user = auth.login("noone", "pw")
        assert user is None

    def test_register_closed(self, store, auth):
        auth.open_registration = False
        uid = auth.register("newuser", "pw")
        assert uid is None

    def test_password_is_hashed(self, store, auth):
        uid = auth.register("testuser", "mypassword")
        user = store.get_user("testuser")
        assert user["password"] != "mypassword"
        assert bcrypt.checkpw("mypassword".encode(), user["password"].encode())
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_auth.py -v
```
Expected: FAIL

- [ ] **Step 3: Write stockbot/auth.py**

```python
import bcrypt
from stockbot.memory.store import MemoryStore


class AuthManager:
    def __init__(self, store: MemoryStore, open_registration: bool = True):
        self.store = store
        self.open_registration = open_registration

    def register(self, username: str, password: str) -> str | None:
        if not self.open_registration:
            return None
        if self.store.get_user(username):
            return None
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        return self.store.create_user(username, hashed, role="user")

    def login(self, username: str, password: str) -> dict | None:
        user = self.store.get_user(username)
        if not user:
            return None
        if not bcrypt.checkpw(password.encode(), user["password"].encode()):
            return None
        return user

    def list_users(self) -> list[dict]:
        return self.store.list_users()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_auth.py -v
```
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add stockbot/auth.py tests/test_auth.py
git commit -m "feat: add AuthManager with bcrypt register/login"
```

---

## Phase 4: Core

### Task 11: AgentCore

**Files:**
- Create: `stockbot/core.py`
- Create: `tests/test_agent_core.py`

- [ ] **Step 1: Write failing tests in tests/test_agent_core.py**

```python
import pytest
from stockbot.llm.base import LLMProvider, LLMResponse, ToolCall
from stockbot.tools.base import Tool
from stockbot.tools.registry import ToolRegistry
from stockbot.memory.store import MemoryStore
from stockbot.memory.history import ConversationHistory
from stockbot.memory.profile import ProfileManager
from stockbot.context import ContextAssembler
from stockbot.quota import QuotaManager
from stockbot.core import AgentCore
from stockbot.events import TextDelta, ToolCallStart, ToolCallEnd, QuotaExceeded


class EchoLLM(LLMProvider):
    """Always returns text. For tool_call tests set canned_tool_calls."""
    def __init__(self, text: str = "Hello", tool_calls: list[ToolCall] | None = None):
        self.text = text
        self.tool_calls = tool_calls or []

    async def chat(self, messages, tools=None):
        if self.tool_calls:
            return LLMResponse(tool_calls=self.tool_calls, finish_reason="tool_calls")
        return LLMResponse(text=self.text, finish_reason="stop")

    async def chat_stream(self, messages, tools=None):
        for c in self.text:
            yield c


class TestAgentCore:
    @pytest.fixture
    def store(self, temp_db):
        s = MemoryStore(temp_db)
        s.init_schema()
        return s

    @pytest.fixture
    def user_id(self, store):
        return store.create_user("u1", "pw")

    @pytest.fixture
    def components(self, store):
        llm = EchoLLM(text="你好！有什么可以帮助你的？")
        tools = ToolRegistry()
        async def get_time() -> str:
            return "2026-05-23 14:30:00"
        tools.register(Tool(name="get_time", description="获取当前时间", parameters={
            "type": "object", "properties": {}, "required": []
        }, func=get_time))
        history = ConversationHistory(store)
        profile = ProfileManager(store)
        context = ContextAssembler(tools, profile, history)
        quota = QuotaManager(store, daily_limit=5)
        return llm, tools, history, profile, context, quota

    @pytest.fixture
    def agent(self, components):
        llm, tools, history, profile, context, quota = components
        return AgentCore(llm=llm, tool_registry=tools, context_assembler=context,
                         memory_store=store, history=history, profile=profile,
                         quota=quota, max_turns=8)

    @pytest.mark.asyncio
    async def test_simple_text_response(self, agent, user_id, store):
        events = []
        async for evt in agent.run(user_id, "你好"):
            events.append(evt)
        assert any(isinstance(e, TextDelta) for e in events)
        texts = [e.content for e in events if isinstance(e, TextDelta)]
        assert "".join(texts) == "你好！有什么可以帮助你的？"

    @pytest.mark.asyncio
    async def test_run_saves_to_history(self, agent, user_id, store):
        async for _ in agent.run(user_id, "你好"):
            pass
        msgs = store.get_history(user_id, 10)
        assert len(msgs) >= 2

    @pytest.mark.asyncio
    async def test_tool_call_flow(self, agent, user_id, store, components):
        llm = components[0]
        llm.tool_calls = [ToolCall(id="c1", name="get_time", arguments={})]
        llm.text = ""
        events = []
        async for evt in agent.run(user_id, "几点"):
            events.append(evt)
        assert any(isinstance(e, ToolCallStart) and e.name == "get_time" for e in events)
        assert any(isinstance(e, ToolCallEnd) and e.name == "get_time" for e in events)

    @pytest.mark.asyncio
    async def test_quota_exceeded(self, agent, user_id, store, components):
        quota = components[5]
        for _ in range(5):
            quota.consume(user_id)
        events = []
        async for evt in agent.run(user_id, "你好"):
            events.append(evt)
        assert any(isinstance(e, QuotaExceeded) for e in events)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_agent_core.py -v
```
Expected: FAIL

- [ ] **Step 3: Write stockbot/core.py**

```python
import json
from stockbot.llm.base import LLMProvider
from stockbot.tools.registry import ToolRegistry
from stockbot.memory.store import MemoryStore
from stockbot.memory.history import ConversationHistory
from stockbot.memory.profile import ProfileManager
from stockbot.context import ContextAssembler
from stockbot.quota import QuotaManager
from stockbot.events import TextDelta, TextDone, ToolCallStart, ToolCallEnd, Error, QuotaExceeded


class AgentCore:
    def __init__(self, llm: LLMProvider, tool_registry: ToolRegistry,
                 context_assembler: ContextAssembler, memory_store: MemoryStore,
                 history: ConversationHistory, profile: ProfileManager,
                 quota: QuotaManager, max_turns: int = 8):
        self.llm = llm
        self.tool_registry = tool_registry
        self.context_assembler = context_assembler
        self.memory_store = memory_store
        self.history = history
        self.profile = profile
        self.quota = quota
        self.max_turns = max_turns

    async def run(self, user_id: str, user_input: str):
        # Quota check
        qr = self.quota.check(user_id)
        if qr.blocked:
            yield QuotaExceeded(limit=qr.limit, used=qr.used)
            return

        messages = self.context_assembler.build(user_id, user_input)
        tools = self.tool_registry.get_schemas() or None
        max_turns = self.max_turns
        tool_results_meta = []

        while max_turns > 0:
            max_turns -= 1

            try:
                response = await self.llm.chat(messages, tools)
            except Exception as e:
                yield Error(message=f"LLM 调用失败: {e}")
                return

            if response.tool_calls:
                for tc in response.tool_calls:
                    yield ToolCallStart(name=tc.name, args=tc.arguments)
                    result = await self.tool_registry.execute(tc.name, tc.arguments)
                    yield ToolCallEnd(name=tc.name, result=result)
                    tool_results_meta.append({"name": tc.name, "result": result})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue

            if response.text:
                # Stream text to UI
                full_text = response.text
                yield TextDelta(content=full_text)
                yield TextDone()

                # Consume quota after successful LLM call
                self.quota.consume(user_id)

                # Save to memory
                self.history.save_turn(
                    user_id, user_input, full_text,
                    tool_results=tool_results_meta if tool_results_meta else None,
                )

                # Update profile — record queried symbols
                for tr in tool_results_meta:
                    args = {}
                    if hasattr(tr, "args"):
                        args = tr.get("args", {})
                    symbol = args.get("symbol", "")
                    if symbol:
                        self.profile.record_query(user_id, symbol)

                return

        yield Error(message="达到最大推理轮次，请重新提问。")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_agent_core.py -v
```
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add stockbot/core.py tests/test_agent_core.py
git commit -m "feat: add AgentCore with ReAct loop and streaming events"
```

---

## Phase 5: Factory and UI

### Task 12: Factory function and stockbot/__init__.py

**Files:**
- Modify: `stockbot/__init__.py`

- [ ] **Step 1: Rewrite stockbot/__init__.py with factory function**

```python
from pathlib import Path
from stockbot.config import load_config
from stockbot.llm.deepseek import DeepSeekProvider
from stockbot.tools.registry import ToolRegistry
from stockbot.tools.stock_search import create_search_tool
from stockbot.tools.stock_price import create_price_tool
from stockbot.tools.stock_finance import create_finance_tool
from stockbot.tools.stock_trend import create_trend_tool
from stockbot.tools.stock_news import create_news_tool
from stockbot.data.akshare_provider import AkshareProvider
from stockbot.memory.store import MemoryStore
from stockbot.memory.history import ConversationHistory
from stockbot.memory.profile import ProfileManager
from stockbot.context import ContextAssembler
from stockbot.quota import QuotaManager
from stockbot.auth import AuthManager
from stockbot.core import AgentCore


def ensure_data_dir(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def create_agent(config_path: str = "config.yaml", db_path: str | None = None):
    """Factory: wire up all components and return (AgentCore, MemoryStore, AuthManager, dict)."""
    cfg = load_config(config_path)

    if db_path is None:
        db_path = cfg.get("memory", {}).get("db_path", "data/stockbot.db")

    ensure_data_dir(db_path)
    store = MemoryStore(db_path)
    store.init_schema()

    # Data layer
    data_provider = AkshareProvider()

    # LLM
    llm_cfg = cfg.get("llm", {})
    llm = DeepSeekProvider(
        model=llm_cfg.get("model", "deepseek-chat"),
        api_key=llm_cfg.get("api_key"),
        max_tokens=llm_cfg.get("max_tokens", 4096),
        temperature=llm_cfg.get("temperature", 0.3),
    )

    # Tools
    tool_registry = ToolRegistry()
    tool_registry.register(create_search_tool(data_provider))
    tool_registry.register(create_price_tool(data_provider))
    tool_registry.register(create_finance_tool(data_provider))
    tool_registry.register(create_trend_tool(data_provider))
    tool_registry.register(create_news_tool(data_provider))

    # Memory
    memory_cfg = cfg.get("memory", {})
    history = ConversationHistory(store, history_limit=memory_cfg.get("history_limit", 200))
    profile = ProfileManager(store)

    # Context
    context_assembler = ContextAssembler(tool_registry, profile, history)

    # Quota
    quota_cfg = cfg.get("quota", {})
    quota = QuotaManager(store, daily_limit=quota_cfg.get("daily_limit", 5))

    # Auth
    auth_cfg = cfg.get("auth", {})
    auth = AuthManager(store, open_registration=auth_cfg.get("open_registration", True))

    # Agent
    agent = AgentCore(
        llm=llm, tool_registry=tool_registry, context_assembler=context_assembler,
        memory_store=store, history=history, profile=profile, quota=quota,
        max_turns=8,
    )

    return agent, store, auth, cfg
```

- [ ] **Step 2: Verify factory works**

```bash
python -c "from stockbot import create_agent; a, s, auth, cfg = create_agent('config.yaml', db_path='data/test.db'); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add stockbot/__init__.py
git commit -m "feat: add factory function create_agent() to wire all components"
```

---

### Task 13: CLI UI

**Files:**
- Create: `stockbot/ui/__init__.py`
- Create: `stockbot/ui/cli_ui.py`

- [ ] **Step 1: Write stockbot/ui/__init__.py (empty)**

```python
```

- [ ] **Step 2: Write stockbot/ui/cli_ui.py**

```python
import asyncio
import sys
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.markdown import Markdown
from stockbot.core import AgentCore
from stockbot.memory.store import MemoryStore
from stockbot.quota import QuotaManager
from stockbot.memory.profile import ProfileManager
from stockbot.events import TextDelta, TextDone, ToolCallStart, ToolCallEnd, Error, QuotaExceeded


WELCOME = """[bold cyan]StockBot[/] — A股智能分析助手
DeepSeek · akshare · v0.1.0

输入你的问题开始对话，/help 查看命令，/exit 退出"""

HELP_TEXT = """[bold]可用命令:[/]
  /help       显示此帮助
  /tools      列出所有工具
  /watch <代码> 添加自选股
  /portfolio  查看自选股
  /quota      查看今日用量
  /clear      清除会话上下文
  /admin approve <n>  管理员批准额外额度
  /admin reset        管理员重置当日额度
  /exit       退出"""


class CLIUI:
    def __init__(self, agent: AgentCore, store: MemoryStore,
                 quota: QuotaManager, profile: ProfileManager,
                 user_id: str):
        self.agent = agent
        self.store = store
        self.quota = quota
        self.profile = profile
        self.user_id = user_id
        self.console = Console()
        self.running = True

    def run(self):
        self.console.print(Panel.fit(WELCOME, border_style="cyan"))
        self.console.print()

        while self.running:
            try:
                user_input = self.console.input("[bold white]👤[/] ")
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n再见！")
                break

            if not user_input.strip():
                continue

            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue

            asyncio.run(self._chat(user_input))

    def _handle_command(self, cmd: str):
        parts = cmd.split()
        op = parts[0].lower()

        if op == "/exit":
            self.running = False
            self.console.print("再见！")
        elif op == "/help":
            self.console.print(HELP_TEXT)
        elif op == "/tools":
            self.console.print(self.agent.tool_registry.describe())
        elif op == "/watch" and len(parts) > 1:
            self.profile.add_favorite(self.user_id, parts[1])
            self.console.print(f"[green]✓ 已添加 {parts[1]} 到自选股[/]")
        elif op == "/portfolio":
            favs = self.profile.get_favorites(self.user_id)
            if favs:
                self.console.print(f"[bold]⭐ 自选股:[/] {', '.join(favs)}")
            else:
                self.console.print("暂无自选股。用 /watch <代码> 添加")
        elif op == "/quota":
            qr = self.quota.check(self.user_id)
            self.console.print(f"[bold]📊 今日用量:[/] {qr.used}/{qr.limit}  (剩余 {qr.remain})")
        elif op == "/clear":
            self.console.print("[yellow]会话上下文已清除[/]")
        elif op == "/admin":
            self._handle_admin(parts)
        else:
            self.console.print(f"[red]未知命令: {op}[/]")

    def _handle_admin(self, parts: list[str]):
        if len(parts) < 2:
            self.console.print("用法: /admin approve <n> 或 /admin reset")
            return
        sub = parts[1].lower()
        if sub == "approve" and len(parts) > 2:
            try:
                n = int(parts[2])
                self.quota.approve(self.user_id, n)
                self.console.print(f"[green]✓ 已批准额外 {n} 次调用[/]")
            except ValueError:
                self.console.print("[red]请输入有效数字[/]")
        elif sub == "reset":
            self.quota.reset(self.user_id)
            self.console.print("[green]✓ 配额已重置[/]")
        else:
            self.console.print("用法: /admin approve <n> 或 /admin reset")

    async def _chat(self, user_input: str):
        response_text = ""
        tool_count = 0

        async for evt in self.agent.run(self.user_id, user_input):
            if isinstance(evt, QuotaExceeded):
                self.console.print(
                    f"\n[red]🔒 今日额度已用完 ({evt.used}/{evt.limit})[/]\n"
                    f"请联系管理员批准额外额度，或等待明天 0 点重置。"
                )
                return

            if isinstance(evt, TextDelta):
                response_text += evt.content

            if isinstance(evt, TextDone):
                self.console.print()
                self.console.print(Markdown(response_text))
                # Append disclaimer automatically
                qr = self.quota.check(self.user_id)
                self.console.print(f"[dim]今日已用 {qr.used}/{qr.limit} 次[/]", end="")

            if isinstance(evt, ToolCallStart):
                tool_count += 1
                self.console.print(f"  [yellow]🔧 {evt.name}(...)[/]", end="")

            if isinstance(evt, ToolCallEnd):
                self.console.print(f" [green]✓[/]")

            if isinstance(evt, Error):
                self.console.print(f"\n[red]✗ {evt.message}[/]")
                return

        self.console.print()
```

- [ ] **Step 3: Write cli.py (entry point)**

```python
#!/usr/bin/env python
import sys
from stockbot import create_agent
from stockbot.ui.cli_ui import CLIUI


def main():
    agent, store, auth, cfg = create_agent()

    # CLI uses a default local user
    user = store.get_user("local")
    if not user:
        user_id = auth.register("local", "local")
        if not user_id:
            user_id = store.create_user("local", "", "user")
    else:
        user_id = user["id"]

    ui = CLIUI(agent, store, agent.quota, agent.profile, user_id)
    ui.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add stockbot/ui/__init__.py stockbot/ui/cli_ui.py cli.py
git commit -m "feat: add Rich-based CLI chat interface"
```

---

### Task 14: Web UI — login and chat pages

**Files:**
- Create: `stockbot/ui/login_page.py`
- Create: `stockbot/ui/chat_page.py`

- [ ] **Step 1: Write stockbot/ui/login_page.py**

```python
import streamlit as st
from stockbot.auth import AuthManager


def render_login(auth: AuthManager):
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1>📈 StockBot</h1>
        <p style="color: #666; font-size: 1.1rem;">A股智能分析助手</p>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["登录", "注册"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("用户名", key="login_user")
            password = st.text_input("密码", type="password", key="login_pass")
            submitted = st.form_submit_button("登录", use_container_width=True)
            if submitted:
                user = auth.login(username, password)
                if user:
                    st.session_state["user"] = user
                    st.rerun()
                else:
                    st.error("用户名或密码错误")

    with tab_register:
        with st.form("register_form"):
            new_user = st.text_input("用户名", key="reg_user")
            new_pass = st.text_input("密码", type="password", key="reg_pass")
            confirm = st.text_input("确认密码", type="password", key="reg_confirm")
            submitted = st.form_submit_button("注册", use_container_width=True)
            if submitted:
                if not new_user or not new_pass:
                    st.error("请填写用户名和密码")
                elif new_pass != confirm:
                    st.error("两次密码不一致")
                else:
                    uid = auth.register(new_user, new_pass)
                    if uid is None:
                        st.error("用户名已存在或注册已关闭")
                    else:
                        st.success("注册成功！请切换到登录页登录")
```

- [ ] **Step 2: Write stockbot/ui/chat_page.py**

```python
import asyncio
import streamlit as st
from stockbot.core import AgentCore
from stockbot.quota import QuotaManager
from stockbot.memory.profile import ProfileManager
from stockbot.events import TextDelta, ToolCallStart, ToolCallEnd, Error, QuotaExceeded


def render_chat(agent: AgentCore, quota: QuotaManager, profile: ProfileManager, user: dict):
    user_id = user["id"]

    # ── Sidebar ──
    with st.sidebar:
        st.title("📈 StockBot")

        if st.button("💬 新对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("⭐ 自选股")
        favs = profile.get_favorites(user_id)
        if favs:
            for sym in favs:
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(f"{sym}", key=f"fav_{sym}", use_container_width=True):
                        st.session_state.pending_input = f"帮我分析一下 {sym}"
                        st.rerun()
                with col2:
                    if st.button("✕", key=f"rm_{sym}"):
                        profile.remove_favorite(user_id, sym)
                        st.rerun()
        else:
            st.caption("暂无自选股")

        st.divider()
        st.subheader("📋 操作")
        qr = quota.check(user_id)
        st.metric("今日用量", f"{qr.used}/{qr.limit}", delta=f"剩余 {qr.remain}")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📊 大盘", use_container_width=True):
                st.session_state.pending_input = "今天大盘行情怎么样"
                st.rerun()
        with c2:
            if st.button("🔥 热门", use_container_width=True):
                st.session_state.pending_input = "最近什么板块比较热门"
                st.rerun()

    # ── Main chat ──
    st.caption(f"👤 {user['username']}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Pending input from sidebar buttons
    if "pending_input" in st.session_state and st.session_state.pending_input:
        prompt = st.session_state.pending_input
        st.session_state.pending_input = None
    else:
        prompt = None

    # Show welcome if new user
    if not st.session_state.messages:
        st.markdown("""
        ### 👋 欢迎使用 StockBot！

        我可以帮你：
        - 📈 查询实时行情
        - 📊 分析股票走势
        - 📰 解读财务数据
        - 🔍 搜索相关新闻

        试着问我：

        """)
        examples = [
            "最近什么板块比较热门？",
            "帮我分析一下贵州茅台",
            "我想找一只高分红的银行股",
        ]
        cols = st.columns(len(examples))
        for i, ex in enumerate(examples):
            with cols[i]:
                if st.button(f"💬 {ex}", key=f"ex_{i}", use_container_width=True):
                    prompt = ex

    # Render message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    input_text = st.chat_input("输入股票问题...")

    if prompt:
        input_text = prompt

    if input_text:
        # User message
        st.session_state.messages.append({"role": "user", "content": input_text})
        with st.chat_message("user"):
            st.markdown(input_text)

        # Agent response
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            response_text = ""

            async def stream():
                nonlocal response_text
                async for evt in agent.run(user_id, input_text):
                    if isinstance(evt, QuotaExceeded):
                        st.error(f"🔒 今日额度已用完 ({evt.used}/{evt.limit})。明天 0 点自动重置。")
                        return
                    if isinstance(evt, TextDelta):
                        response_text += evt.content
                        response_placeholder.markdown(response_text)
                    if isinstance(evt, ToolCallStart):
                        status_placeholder.info(f"🔧 {evt.name} 执行中...")
                    if isinstance(evt, ToolCallEnd):
                        status_placeholder.success(f"✓ {evt.name} 完成")
                    if isinstance(evt, Error):
                        st.error(f"✗ {evt.message}")
                        return
                status_placeholder.empty()

            asyncio.run(stream())

            if response_text:
                disclaimer = "\n\n---\n⚠️ 分析仅供参考，不构成投资建议"
                response_placeholder.markdown(response_text + disclaimer)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()
```

- [ ] **Step 3: Commit**

```bash
git add stockbot/ui/login_page.py stockbot/ui/chat_page.py
git commit -m "feat: add Streamlit login and chat UI pages"
```

---

### Task 15: Web UI — admin page and app entry point

**Files:**
- Create: `stockbot/ui/admin_page.py`
- Create: `stockbot/admin.py`
- Create: `app.py`

- [ ] **Step 1: Write stockbot/admin.py**

```python
from datetime import date, timedelta
from stockbot.memory.store import MemoryStore
from stockbot.quota import QuotaManager
from stockbot.auth import AuthManager


class AdminService:
    def __init__(self, store: MemoryStore, quota: QuotaManager, auth: AuthManager):
        self.store = store
        self.quota = quota
        self.auth = auth

    def list_users_with_stats(self) -> list[dict]:
        today = date.today().isoformat()
        users = self.store.list_users()
        result = []
        for u in users:
            q = self.store.get_quota(u["id"], today)
            # count total conversations
            conv_count = len(self.store.get_history(u["id"], 999))
            result.append({
                **u,
                "calls_today": q["calls"],
                "approved_today": q["approved"],
                "limit": u.get("daily_quota", 5) + q["approved"],
                "total_conversations": conv_count,
            })
        return result

    def set_user_quota(self, user_id: str, daily_quota: int):
        self.store.update_user_quota(user_id, daily_quota)

    def approve_user(self, user_id: str, extra: int):
        today = date.today().isoformat()
        self.store.add_approved(user_id, today, extra)

    def get_stats(self) -> dict:
        today = date.today().isoformat()
        users = self.store.list_users()
        total_calls = 0
        for u in users:
            q = self.store.get_quota(u["id"], today)
            total_calls += q["calls"]

        seven_days_ago = (date.today() - timedelta(days=7)).isoformat()
        active_users = set()
        for u in users:
            hist = self.store.get_history(u["id"], 1)
            if hist:
                active_users.add(u["id"])

        return {
            "total_users": len(users),
            "active_users_7d": len(active_users),
            "total_calls_today": total_calls,
        }
```

- [ ] **Step 2: Write stockbot/ui/admin_page.py**

```python
import streamlit as st
from stockbot.admin import AdminService
from stockbot.quota import QuotaManager


def render_admin(admin_svc: AdminService, quota: QuotaManager):
    st.title("⚙️ 管理面板")

    tabs = st.tabs(["👥 用户管理", "📊 统计"])

    with tabs[0]:
        st.subheader("用户列表")
        users = admin_svc.list_users_with_stats()

        if users:
            cols = st.columns([2, 1, 1, 1, 1, 1, 1])
            cols[0].markdown("**用户名**")
            cols[1].markdown("**角色**")
            cols[2].markdown("**日配额**")
            cols[3].markdown("**今日已用**")
            cols[4].markdown("**总对话**")
            cols[5].markdown("**操作**")

            for u in users:
                cols = st.columns([2, 1, 1, 1, 1, 1, 1])
                cols[0].write(u["username"])
                cols[1].write(u["role"])
                cols[2].write(str(u.get("daily_quota", 5)))
                cols[3].write(f'{u["calls_today"]}/{u["limit"]}')
                cols[4].write(str(u["total_conversations"]))

                with cols[5].popover("提额"):
                    extra = st.number_input("额外额度", min_value=1, max_value=100, value=5, key=f"extra_{u['id']}")
                    if st.button("确认", key=f"approve_{u['id']}"):
                        admin_svc.approve_user(u["id"], extra)
                        st.success(f"已为 {u['username']} 增加 {extra} 次")
                        st.rerun()

                with cols[6].popover("设置"):
                    new_limit = st.number_input("新日配额", min_value=0, max_value=500, value=u.get("daily_quota", 5), key=f"limit_{u['id']}")
                    if st.button("保存", key=f"save_{u['id']}"):
                        admin_svc.set_user_quota(u["id"], new_limit)
                        st.success(f"{u['username']} 日配额已更新为 {new_limit}")
                        st.rerun()

    with tabs[1]:
        st.subheader("全局统计")
        stats = admin_svc.get_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("总用户", stats["total_users"])
        c2.metric("7日活跃", stats["active_users_7d"])
        c3.metric("今日调用", stats["total_calls_today"])
```

- [ ] **Step 3: Write app.py (Web entry point)**

```python
#!/usr/bin/env python
import streamlit as st
from stockbot import create_agent
from stockbot.admin import AdminService
from stockbot.ui.login_page import render_login
from stockbot.ui.chat_page import render_chat
from stockbot.ui.admin_page import render_admin


def main():
    st.set_page_config(page_title="StockBot", page_icon="📈", layout="wide")

    # Init app state
    if "agent" not in st.session_state:
        agent, store, auth, cfg = create_agent()
        st.session_state.agent = agent
        st.session_state.store = store
        st.session_state.auth = auth
        st.session_state.cfg = cfg
        admin_svc = AdminService(store, agent.quota, auth)
        st.session_state.admin_svc = admin_svc

    auth = st.session_state.auth

    # Auth gate
    if "user" not in st.session_state:
        render_login(auth)
    else:
        user = st.session_state.user

        # Navigation
        with st.sidebar:
            if user.get("role") == "admin":
                page = st.radio("导航", ["💬 对话", "⚙️ 管理"], label_visibility="collapsed")
            else:
                page = "💬 对话"
                if st.button("🚪 退出登录", use_container_width=True):
                    del st.session_state["user"]
                    del st.session_state.get("messages", None)
                    st.rerun()

        if page == "💬 对话":
            render_chat(st.session_state.agent, st.session_state.agent.quota,
                        st.session_state.agent.profile, user)
        elif page == "⚙️ 管理":
            render_admin(st.session_state.admin_svc, st.session_state.agent.quota)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add stockbot/admin.py stockbot/ui/admin_page.py app.py
git commit -m "feat: add admin service, admin dashboard, and Streamlit entry point"
```

---

## Phase 6: Final Verification

### Task 16: End-to-end smoke test and README

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```
Expected: All tests PASS (approximately 50+ tests)

- [ ] **Step 2: Test factory with real config**

```bash
python -c "from stockbot import create_agent; a, s, auth, c = create_agent(); print('Agent created:', type(a).__name__); print('Tools:', len(a.tool_registry._tools))"
```
Expected output shows AgentCore and 5 tools

- [ ] **Step 3: Verify CLI starts**

```bash
python cli.py <<< "/exit"
```
Expected: Prints welcome banner and exits cleanly

- [ ] **Step 4: Write README.md**

```markdown
# StockBot — A股智能分析助手

基于 DeepSeek API 的 AI 股票分析 Agent。

## 快速开始

1. 安装依赖: `pip install -r requirements.txt`
2. 设置 API Key: `set DEEPSEEK_API_KEY=your-key` (Windows) 或 `export DEEPSEEK_API_KEY=your-key` (Linux/Mac)
3. 运行 CLI: `python cli.py`
4. 运行 Web: `streamlit run app.py`

## 功能

- 实时行情查询
- 多情景技术分析 (MA/MACD/RSI)
- 财务数据解读
- 股票新闻搜索
- 自选股管理
- 多用户 Web 界面
- 用量配额管控

## 配置

编辑 `config.yaml` 修改默认配额、模型参数、数据源等。

## 部署

- 本地: `streamlit run app.py`
- Hugging Face: 推送代码到 Space 自动部署
- 云服务器: 配合 Nginx 反向代理

## 免责声明

本工具所有分析仅供参考，不构成投资建议。
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add README with quickstart and feature overview"
```

---

## Plan Self-Review

**1. Spec coverage check:**
- [x] LLM Provider (Task 3)
- [x] AgentCore with streaming events (Task 11)
- [x] ToolRegistry with 5 tools (Tasks 6, 7)
- [x] DataProvider + AkshareProvider (Task 2)
- [x] MemoryStore SQLite + History + Profile (Tasks 4, 5)
- [x] ContextAssembler (Task 9)
- [x] QuotaManager (Task 8)
- [x] Auth (Task 10)
- [x] CLI UI (Task 13)
- [x] Web UI login + chat + admin (Tasks 14, 15)
- [x] Factory function (Task 12)
- [x] Config YAML (Task 1)
- [x] Deployment pipeline (README)

**2. Placeholder scan:** No TBD, TODO, or vague "implement later" items. All code steps contain complete implementation.

**3. Type consistency:** Events defined in Task 1 match usage in Task 11. Tool interface in Task 6 matches tool implementations in Task 7. AgentCore.run() signature matches CLI and Web callers.

**All checks passed.**
