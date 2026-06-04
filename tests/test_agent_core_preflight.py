from __future__ import annotations

import pytest

from agent_core.preflight import (
    AgentRunPreflightCenter,
    AgentRunPreflightRequest,
    AgentRunPreflightRequirements,
    StaticRunPreflightCheck,
)


@pytest.mark.asyncio
async def test_static_run_preflight_reports_missing_requirements_and_task_budget() -> None:
    center = AgentRunPreflightCenter((StaticRunPreflightCheck(),))

    report = await center.check(
        AgentRunPreflightRequest(
            task="scan target",
            session_name="sdk",
            requirements=AgentRunPreflightRequirements(
                max_task_bytes=4,
                required_actions=("finish",),
                required_tools=("scan",),
                required_skills=("review",),
                required_mcp_servers=("fs",),
                require_memory=True,
            ),
            available_actions=("finish",),
            available_tools=("lookup",),
            available_skills=(),
            available_mcp_servers=(),
            memory_enabled=False,
        )
    )
    manifest = report.manifest()

    assert report.status == "blocked"
    assert not report.ok
    assert manifest["blocking_count"] == 5
    assert manifest["blocking_codes"] == [
        "memory_required",
        "missing_mcp_server",
        "missing_skill",
        "missing_tool",
        "task_bytes_exceeded",
    ]
    assert manifest["request"]["task_bytes"] == len("scan target".encode("utf-8"))


@pytest.mark.asyncio
async def test_static_run_preflight_allows_warnings_without_blocking() -> None:
    report = await AgentRunPreflightCenter.default().check(AgentRunPreflightRequest(task=""))

    assert report.status == "passed"
    assert report.ok
    assert report.manifest()["codes"] == ["empty_task"]
