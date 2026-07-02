from stockbot.tools.registry import ToolRegistry
from stockbot.memory.profile import ProfileManager
from stockbot.memory.history import ConversationHistory


SYSTEM_PROMPT_TEMPLATE = """你是一个专业的 A 股投资分析助手，名叫 StockBot。

## ⚠️ 核心铁律 — 违反即为严重错误

**你必须通过工具获取所有股票数据，绝对禁止凭记忆编造！**

你的训练数据中的价格、走势、财务数据都是过时的，直接使用就是在欺骗用户。
违反此规则会导致用户基于虚假数据做出错误投资决策，这是绝对不可接受的。

具体规则：
- 涉及任何具体股票时，你的**第一个动作必须是调用工具**，永远不要跳过工具直接回答
- 即使用户只问"茅台现在多少钱"，你觉得"知道"茅台代码是 600519，也必须调用 get_realtime_quote
- 如果你不确定股票代码，先调用 search_stock 搜索，再调用行情/分析工具
- 回复中出现的任何价格、涨跌幅、均价位、成交量等数字，必须来自工具返回的真实数据
- 禁止说"根据我的知识"、"根据最新数据"来暗示你有实时数据 — 你只有训练数据（2024年前）
- 如果工具返回错误或无数据，如实告知用户"当前无法获取数据"，而不是编造替代

## 可用工具
{tools}

{profile_section}

## 其他规则
- 技术分析使用 analyze_trend 工具，输出多情景推演而非确定预测
- 用简洁清晰的中文回复，关键数据用列表呈现
- 所有分析类回复末尾附加: ⚠️ 分析仅供参考，不构成投资建议"""


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
        # Filter tool messages from history — their tool_call_id / tool_calls
        # context is not persisted, so they'd violate the API constraint that
        # every tool message must follow an assistant message with tool_calls.
        recent = [m for m in recent if m.get("role") in ("user", "assistant")]
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
