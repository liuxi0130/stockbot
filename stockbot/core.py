import json
import re

from stockbot.llm.base import LLMProvider
from stockbot.tools.registry import ToolRegistry
from stockbot.memory.store import MemoryStore
from stockbot.memory.history import ConversationHistory
from stockbot.memory.profile import ProfileManager
from stockbot.context import ContextAssembler
from stockbot.quota import QuotaManager
from stockbot.events import TextDelta, TextDone, ToolCallStart, ToolCallEnd, Error, QuotaExceeded


# ── Stock query detection (two tiers) ──────────────────
# Tier 1: Stock code pattern — unambiguous, triggers API-level tool_choice="required"
_STOCK_CODE_RE = re.compile(r'\b\d{6}\b')

# Tier 2: Stock-related keywords — triggers reactive guard for name-based queries
# (e.g. "茅台多少钱" without typing the code)
_STOCK_DATA_KEYWORDS = [
    '股价', '价格', '多少钱', '行情', '涨跌', '涨幅', '跌幅',
    '涨了', '跌了', '上涨', '下跌',
    '现在', '今天', '最近', '当前', '实时', '最新',
    '收盘', '开盘', '最高', '最低', '成交量', '换手率', '市值', '市盈率',
    'PE', 'PB', 'ROE', 'EPS',
    '大盘', '指数', '上证', '深证', '创业板', '沪深',
    '走势', '趋势', '技术分析', '均线',
]


def _has_stock_code(text: str) -> bool:
    """Return True if the text contains a 6-digit stock code pattern."""
    return bool(_STOCK_CODE_RE.search(text))


def _is_stock_data_query(text: str) -> bool:
    """Return True if the user input is asking for stock data that requires tools."""
    if _has_stock_code(text):
        return True
    if any(kw in text for kw in _STOCK_DATA_KEYWORDS):
        return True
    return False


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
        # Log user query for admin audit
        self.memory_store.log_activity(user_id, "query", user_input[:200])

        qr = self.quota.check(user_id)
        if qr.blocked:
            yield QuotaExceeded(limit=qr.limit, used=qr.used)
            return

        messages = self.context_assembler.build(user_id, user_input)
        tools = self.tool_registry.get_schemas() or None
        max_turns = self.max_turns
        tool_results_meta = []

        # ── Structural guard: if user typed a stock code, force tool call at API level ──
        force_tools = bool(tools and _has_stock_code(user_input))

        while max_turns > 0:
            max_turns -= 1

            # API-level enforcement: "required" forces the LLM to call ≥1 tool.
            # Once tools have executed (tool_results_meta non-empty), relax to auto.
            if force_tools and not tool_results_meta:
                tc = "required"
            else:
                tc = None  # default "auto"

            try:
                response = await self.llm.chat(messages, tools, tool_choice=tc)
            except Exception as e:
                yield Error(message=f"LLM 调用失败: {e}")
                return

            if response.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                })
                for tc in response.tool_calls:
                    yield ToolCallStart(name=tc.name, args=tc.arguments)
                    result = await self.tool_registry.execute(tc.name, tc.arguments)
                    yield ToolCallEnd(name=tc.name, result=result)
                    tool_results_meta.append({"name": tc.name, "result": result})
                    # Log tool activity for admin audit
                    detail = str(tc.arguments.get(list(tc.arguments.keys())[0], "")) if tc.arguments else ""
                    self.memory_store.log_activity(user_id, tc.name, detail)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue

            # ── Guard: block direct text responses that skip tools on stock queries ──
            if (response.text
                    and not tool_results_meta
                    and _is_stock_data_query(user_input)):
                # LLM tried to answer a stock-data question without calling any tools.
                # Don't show the hallucinated response; inject a corrective message
                # and force the LLM back into the reasoning loop.
                messages.append({
                    "role": "user",
                    "content": (
                        "【系统警告】你刚才试图不调用任何工具就直接回答股票问题，"
                        "这违反了核心铁律！你的训练数据是过时的，不能用于提供实时股价。"
                        "请立即调用正确的工具（get_realtime_quote、search_stock 等）"
                        "获取真实数据后重新回答。"
                    ),
                })
                continue

            if response.text:
                full_text = response.text
                yield TextDelta(content=full_text)
                yield TextDone()

                self.quota.consume(user_id)

                self.history.save_turn(
                    user_id, user_input, full_text,
                    tool_results=tool_results_meta if tool_results_meta else None,
                )

                for tr in tool_results_meta:
                    symbol = tr.get("symbol", "")
                    if symbol:
                        self.profile.record_query(user_id, symbol)

                return

        yield Error(message="达到最大推理轮次，请重新提问。")
