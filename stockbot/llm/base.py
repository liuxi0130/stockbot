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
