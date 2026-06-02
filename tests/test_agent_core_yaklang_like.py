from __future__ import annotations

import pytest

from agent_core.actions import ActionRegistry, ActionSpec
from agent_core.skills import SkillRegistry, skill_from_markdown
from agent_core.timeline import TimelineStore
from agent_core.tools import ToolInvocation, ToolRegistry, ToolResult, ToolSpec


def test_action_registry_extracts_json_action_from_wrapped_model_text() -> None:
    registry = ActionRegistry()
    registry.register(ActionSpec(name="finish"))

    parsed = registry.parse('thinking...\n{"action":"finish","arguments":{"output":"ok"}}\nthanks')

    assert parsed.name == "finish"
    assert parsed.arguments == {"output": "ok"}


def test_action_registry_accepts_legacy_next_action_shape() -> None:
    registry = ActionRegistry()
    registry.register(
        ActionSpec(
            name="call_tool",
            parameters_schema={
                "type": "object",
                "required": ["tool_name"],
                "properties": {"tool_name": {"type": "string"}},
            },
        )
    )

    parsed = registry.parse({"next_action": {"type": "call_tool", "arguments": {"tool_name": "x"}}})

    assert parsed.name == "call_tool"
    assert parsed.arguments == {"tool_name": "x"}


@pytest.mark.asyncio
async def test_tool_registry_supports_alias_tags_and_manifest() -> None:
    registry = ToolRegistry()

    async def handler(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            content=f"used {invocation.tool_name}",
        )

    registry.register(
        ToolSpec(
            name="http_probe",
            description="Probe HTTP service",
            aliases=("probe",),
            tags=("web", "recon"),
            parameters_schema={"type": "object"},
        ),
        handler,
    )

    result = await registry.invoke(ToolInvocation(tool_name="probe"))
    manifest = registry.manifest(tags=("web",))

    assert result.ok
    assert result.tool_name == "http_probe"
    assert registry.get("probe").name == "http_probe"  # type: ignore[union-attr]
    assert manifest["tools"][0]["aliases"] == ["probe"]
    assert "tags=web,recon" in registry.render_inventory(tags=("recon",))


def test_skill_registry_loads_markdown_frontmatter() -> None:
    skill = skill_from_markdown(
        """---
name: code-review
description: Review code carefully
tags: [code, review]
priority: 7
---
# Review Skill

Check behavior before style.
"""
    )

    assert skill.name == "code-review"
    assert skill.description == "Review code carefully"
    assert skill.tags == ("code", "review")
    assert skill.priority == 7
    assert "Check behavior" in skill.prompt


def test_skill_registry_can_load_markdown_file(tmp_path) -> None:
    path = tmp_path / "skill.md"
    path.write_text(
        """---
name: recon
tags: web
---
# Recon

Prefer passive checks.
""",
        encoding="utf-8",
    )
    registry = SkillRegistry()

    spec = registry.load_markdown_file(path)

    assert spec.name == "recon"
    assert registry.select("web target")[0].name == "recon"


def test_timeline_supports_truncate_after_and_byte_buckets() -> None:
    timeline = TimelineStore()
    first = timeline.add("first " + "a" * 20)
    middle = timeline.add("middle " + "b" * 20)
    timeline.add("last " + "c" * 20)

    deleted = timeline.truncate_after(middle.item_id)
    buckets = timeline.group_by_bytes(35)

    assert deleted == 1
    assert [item.item_id for item in timeline.items if not item.deleted] == [
        first.item_id,
        middle.item_id,
    ]
    assert len(buckets) == 2

