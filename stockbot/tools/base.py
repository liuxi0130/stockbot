from dataclasses import dataclass
from typing import Callable, Any
import asyncio


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
        result = self.func(**kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return str(result)
