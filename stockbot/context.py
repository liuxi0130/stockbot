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
