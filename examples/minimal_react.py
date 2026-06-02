from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_core.config import AgentProfile, RuntimeBudget
from agent_core.providers import LLMProviderCenter, LLMProviderPort, LLMRequest, LLMResponse
from agent_core.runner import AgentRunner, AgentSession
from agent_core.tools import ToolRegistry


class ScriptedProvider(LLMProviderPort):
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self.responses = [
            {
                "action": "call_tool",
                "arguments": {
                    "tool_name": "summarize_target",
                    "arguments": {"target": "demo-service"},
                },
            },
            {"action": "finish", "arguments": {"output": "demo-service summarized"}},
        ]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(action=self.responses.pop(0))


async def main() -> None:
    provider = ScriptedProvider()
    providers = LLMProviderCenter(default_provider="scripted")
    providers.register("scripted", provider, default_model="scripted-model")

    tools = ToolRegistry()
    tool_calls: list[str] = []

    @tools.register_function(description="Summarize a target name.", tags=("demo",))
    def summarize_target(target: str) -> str:
        tool_calls.append(target)
        return f"target={target}; risk=unknown; next=inspect evidence"

    session = AgentSession(
        profile=AgentProfile(
            name="minimal-react-demo",
            model="scripted-model",
            instructions="Use tools when useful, then finish with a concise result.",
            budget=RuntimeBudget(max_iterations=4),
        ),
        provider=providers,
        tools=tools,
    )

    outcome = await AgentRunner(session).run("summarize the demo service")
    print(json.dumps(
        {
            "status": outcome.result.status,
            "output": outcome.result.output,
            "tool_calls": len(tool_calls),
            "provider_requests": len(provider.requests),
        },
        indent=2,
    ))


if __name__ == "__main__":
    asyncio.run(main())
