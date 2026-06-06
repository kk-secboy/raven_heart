from __future__ import annotations

import json

import pytest

from agent_core.actions import ActionRegistry
from agent_core.config import AgentProfile
from agent_core.context import AgentContextPack
from agent_core.prompt import PromptIR
from agent_core.react import ReActConfig, ReActExecutor
from agent_core.runner import AgentRunner, AgentRunRequest, AgentSession
from agent_core.structured import JsonStructuredOutputValidator, StructuredOutputSpec
from agent_core.testing import InMemoryHarness, MockLLMProvider, MockToolRuntime


def _request_prompt_text(request) -> str:
    return "\n\n".join(
        message.content
        for message in request.messages
        if message.metadata.get("agent_core_prompt")
    )


def _answer_schema() -> dict[str, object]:
    return {
        "type": "object",
        "required": ["summary", "risk"],
        "properties": {
            "summary": {"type": "string"},
            "risk": {"type": "integer"},
        },
    }


@pytest.mark.asyncio
async def test_react_repairs_invalid_structured_finish_output() -> None:
    provider = MockLLMProvider(
        [
            {"action": "finish", "arguments": {"output": "plain text"}},
            {
                "action": "finish",
                "arguments": {"output": '{"summary":"done","risk":2}'},
            },
        ]
    )
    harness = InMemoryHarness()
    spec = StructuredOutputSpec(schema=_answer_schema(), max_repairs=1)
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=MockToolRuntime(),
        action_registry=ActionRegistry(),
        harness=harness,
        config=ReActConfig(max_iterations=3, structured_output=spec),
    )

    result = await executor.run("summarize", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert json.loads(result.output) == {"summary": "done", "risk": 2}
    assert result.metadata["structured_output"]["ok"] is True
    assert result.metadata["structured_output"]["value"]["risk"] == 2
    assert harness.checkpoints[0].state["status"] == "structured_output_error"
    assert harness.checkpoints[-1].state["structured_output"]["ok"] is True
    assert provider.requests[1].messages[-1].content.startswith(
        '{"feedback": "structured_output_error"'
    )


@pytest.mark.asyncio
async def test_structured_output_result_contains_schema_validation_manifest() -> None:
    validator = JsonStructuredOutputValidator()
    spec = StructuredOutputSpec(name="risk_summary", schema=_answer_schema())

    result = await validator.validate('{"summary":"done","risk":2}', spec)
    failed = await validator.validate('{"summary":"done"}', spec)

    assert result.ok
    assert result.metadata["schema_validation"]["schema_name"] == "risk_summary"
    assert result.metadata["schema_validation"]["ok"] is True
    assert not failed.ok
    assert failed.error == "$.risk is required"
    assert failed.metadata["schema_validation"]["issues"][0]["code"] == "required_missing"


@pytest.mark.asyncio
async def test_react_fails_when_structured_output_repairs_are_exhausted() -> None:
    provider = MockLLMProvider(
        [
            {"action": "finish", "arguments": {"output": '{"summary": "missing risk"}'}},
        ]
    )
    harness = InMemoryHarness()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=MockToolRuntime(),
        action_registry=ActionRegistry(),
        harness=harness,
        config=ReActConfig(
            max_iterations=1,
            structured_output=StructuredOutputSpec(schema=_answer_schema(), max_repairs=0),
        ),
    )

    result = await executor.run("summarize", PromptIR.from_parts(dynamic="task"))

    assert result.status == "output_validation_failed"
    assert result.metadata["structured_output"]["error"] == "$.risk is required"
    assert harness.finished[0]["status"] == "output_validation_failed"


@pytest.mark.asyncio
async def test_agent_runner_injects_structured_output_schema_into_prompt() -> None:
    provider = MockLLMProvider(
        [
            {
                "action": "finish",
                "arguments": {"output": '{"summary":"ready","risk":1}'},
            }
        ]
    )
    spec = StructuredOutputSpec(
        name="risk_summary",
        description="Risk summary object.",
        schema=_answer_schema(),
    )
    session = AgentSession(
        profile=AgentProfile(name="structured"),
        provider=provider,
        tools=MockToolRuntime(),
        structured_output_validator=JsonStructuredOutputValidator(),
    )

    outcome = await AgentRunner(session).run(
        AgentRunRequest(
            task="produce risk summary",
            context=AgentContextPack(schema="existing schema note"),
            structured_output=spec,
        )
    )

    prompt_text = _request_prompt_text(provider.requests[0])
    assert outcome.result.status == "completed"
    assert "existing schema note" in prompt_text
    assert "== Structured Output ==" in prompt_text
    assert '"risk"' in prompt_text
    assert provider.requests[0].response_format is not None
    assert provider.requests[0].response_format.kind == "json_schema"
    assert provider.requests[0].response_format.name == "risk_summary"
    assert outcome.prompt_manifest["metadata"]["structured_output"]["name"] == "risk_summary"
    assert outcome.result.metadata["structured_output"]["value"]["summary"] == "ready"
    assert outcome.trace_manifest["summary"]["structured_output_record_count"] == 1
    assert outcome.trace_manifest["summary"]["structured_output_repair_count"] == 0
    assert outcome.trace_manifest["structured_output_trace"]["ok_count"] == 1
    assert outcome.trace_manifest["structured_output_trace"]["schema_names"] == {
        "risk_summary": 1
    }
