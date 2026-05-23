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
