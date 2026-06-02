from __future__ import annotations

import pytest

from agent_core.context import AgentContextPack, AgentPromptBuilder
from agent_core.prompt import PromptBucketRole
from agent_core.skills import (
    SkillRegistry,
    SkillSpec,
    SkillsContext,
    SkillViewWindow,
    select_skills_by_token_budget,
    transform_includes_to_resource_hints,
)
from agent_core.timeline import TimelineBudget, TimelineStore
from agent_core.tools import ToolInvocation, ToolRegistry, ToolResult, ToolSpec


@pytest.mark.asyncio
async def test_prompt_builder_places_tools_skills_timeline_in_expected_buckets() -> None:
    tools = ToolRegistry()

    async def handler(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(call_id=invocation.call_id, tool_name=invocation.tool_name, content="ok")

    tools.register(ToolSpec(name="lookup", description="Lookup data", tags=("recon",)), handler)

    registry = SkillRegistry()
    registry.register(
        SkillSpec(
            name="code-review",
            description="Review code",
            prompt="Check behavior first.",
            tags=("code",),
            priority=5,
        )
    )
    skills = SkillsContext(registry)
    skills.load("code-review")

    timeline = TimelineStore()
    for index in range(5):
        timeline.add(f"observation {index} " + ("x" * 60), kind="tool")

    prompt = AgentPromptBuilder(
        tools=tools,
        skills=skills,
        timeline=timeline,
        timeline_budget=TimelineBudget(max_bytes=160, recent_keep_ratio=0.25),
    ).build(
        AgentContextPack(
            system="system rules",
            task_instruction="follow schema",
            schema='{"type":"object"}',
            output_example='{"ok":true}',
            recent_tools_cache="lookup recently used",
            workspace="linux /tmp",
            current_time="2026-06-02 10:00:00",
            dynamic_task="inspect target",
        )
    )

    assert "system rules" in prompt.bucket(PromptBucketRole.HIGH_STATIC).content
    assert "[tool_inventory]" in prompt.bucket(PromptBucketRole.FROZEN).content
    assert "[timeline_frozen]" in prompt.bucket(PromptBucketRole.FROZEN).content
    assert "[skills_context]" in prompt.bucket(PromptBucketRole.SEMI_DYNAMIC_1).content
    assert "[recent_tools_cache]" in prompt.bucket(PromptBucketRole.SEMI_DYNAMIC_1).content
    assert "[schema]" in prompt.bucket(PromptBucketRole.SEMI_DYNAMIC_2).content
    assert "[timeline_open]" in prompt.bucket(PromptBucketRole.TIMELINE_OPEN).content
    assert "[workspace]" in prompt.bucket(PromptBucketRole.TIMELINE_OPEN).content
    assert "inspect target" in prompt.bucket(PromptBucketRole.DYNAMIC).content


def test_skills_context_renders_loaded_and_available_sections() -> None:
    registry = SkillRegistry()
    registry.register(
        SkillSpec(name="alpha", description="Alpha skill", prompt="Alpha body", priority=10)
    )
    registry.register(SkillSpec(name="beta", description="Beta skill", prompt="Beta body"))
    context = SkillsContext(registry)
    context.load("alpha")

    rendered = context.render_stable()

    assert "== Currently Loaded Skills ==" in rendered
    assert "[skill:alpha]" in rendered
    assert "== Available Skills" in rendered
    assert "  - beta: Beta skill" in rendered


def test_skill_budget_selection_reports_omitted_count() -> None:
    skills = tuple(
        SkillSpec(name=f"skill-{index}", description="x" * 200) for index in range(5)
    )

    listed, omitted = select_skills_by_token_budget(skills, max_tokens=80)

    assert listed
    assert omitted > 0


def test_transform_includes_to_resource_hints() -> None:
    rendered = transform_includes_to_resource_hints(
        "Read this: <!-- include: references/rules.md -->", "review"
    )

    assert "@review/references/rules.md" in rendered
    assert "load_skill_resource" in rendered


def test_skill_view_window_full_and_partial_rendering() -> None:
    full = SkillViewWindow(
        skill_name="review",
        file_path="rules.md",
        content="line one\nline two",
        max_bytes=1024,
    )
    full_rendered, full_truncated = full.render()

    partial = SkillViewWindow(
        skill_name="review",
        file_path="large.md",
        content="\n".join(f"line {index} {'x' * 40}" for index in range(30)),
        max_bytes=260,
    )
    partial.set_offset(3)
    partial_rendered, partial_truncated = partial.render()

    assert not full_truncated
    assert "1 |" not in full_rendered
    assert partial_truncated
    assert "3 | line 2" in partial_rendered

