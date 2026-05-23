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
