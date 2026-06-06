from __future__ import annotations

import pytest

from agent_core.context import (
    AgentContextPack,
    AgentPromptBuilder,
    ContextInjection,
    ContextInjectionPolicy,
    ContextMaterial,
    ContextMaterialSelectionRequest,
    DefaultContextMaterialSelector,
)
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
    assert "[timeline_frozen]" not in prompt.bucket(PromptBucketRole.FROZEN).content
    assert "[skills_context]" in prompt.bucket(PromptBucketRole.SEMI_DYNAMIC_1).content
    assert "[recent_tools_cache]" in prompt.bucket(PromptBucketRole.SEMI_DYNAMIC_1).content
    assert "[skills_context]" not in prompt.bucket(PromptBucketRole.SEMI_DYNAMIC_2).content
    assert "[schema]" in prompt.bucket(PromptBucketRole.SEMI_DYNAMIC_2).content
    assert "[timeline_frozen]" in prompt.bucket(PromptBucketRole.TIMELINE_OPEN).content
    assert "[timeline_open]" in prompt.bucket(PromptBucketRole.TIMELINE_OPEN).content
    assert "[workspace]" in prompt.bucket(PromptBucketRole.TIMELINE_OPEN).content
    assert "inspect target" in prompt.bucket(PromptBucketRole.DYNAMIC).content
    assert prompt.bucket(PromptBucketRole.SEMI_DYNAMIC_1).manifest()["cache_hint"][
        "cacheable"
    ] is False
    assert prompt.bucket(PromptBucketRole.SEMI_DYNAMIC_2).manifest()["cache_hint"][
        "cacheable"
    ] is True


def test_prompt_builder_places_context_injections_by_target_bucket() -> None:
    prompt = AgentPromptBuilder().build(
        AgentContextPack(
            dynamic_task="inspect",
            injections=(
                ContextInjection(
                    name="operator_hint",
                    content="operator says continue",
                    target=PromptBucketRole.DYNAMIC,
                    source="runtime",
                    priority=5,
                    metadata={"request_id": "r1"},
                ),
                ContextInjection(
                    name="resume_checkpoint",
                    content="checkpoint state",
                    target=PromptBucketRole.TIMELINE_OPEN,
                    source="harness",
                    priority=10,
                ),
            ),
        )
    )
    manifest = prompt.manifest()["metadata"]["context_injections"]

    assert "[context_injection:operator_hint source=runtime]" in prompt.bucket(
        PromptBucketRole.DYNAMIC
    ).content
    assert "operator says continue" in prompt.bucket(PromptBucketRole.DYNAMIC).content
    assert "[context_injection:resume_checkpoint source=harness]" in prompt.bucket(
        PromptBucketRole.TIMELINE_OPEN
    ).content
    assert manifest[0]["name"] == "resume_checkpoint"
    assert manifest[1]["name"] == "operator_hint"
    assert manifest[1]["metadata"]["request_id"] == "r1"


def test_prompt_builder_applies_context_injection_policy() -> None:
    prompt = AgentPromptBuilder(
        injection_policy=ContextInjectionPolicy(
            max_injection_bytes=80,
            max_total_bytes=110,
            allowed_targets=(PromptBucketRole.TIMELINE_OPEN, PromptBucketRole.DYNAMIC),
        )
    ).build(
        AgentContextPack(
            dynamic_task="inspect",
            injections=(
                ContextInjection(
                    name="large_hint",
                    content="alpha " * 80,
                    target=PromptBucketRole.DYNAMIC,
                    source="runtime",
                    priority=10,
                ),
                ContextInjection(
                    name="denied_static",
                    content="do not place this in high static",
                    target=PromptBucketRole.HIGH_STATIC,
                    source="runtime",
                    priority=9,
                ),
                ContextInjection(
                    name="overflow",
                    content="beta " * 80,
                    target=PromptBucketRole.TIMELINE_OPEN,
                    source="runtime",
                    priority=1,
                ),
            ),
        )
    )
    manifest = prompt.manifest()["metadata"]["context_injections"]

    assert "[...context injection trimmed...]" in prompt.bucket(PromptBucketRole.DYNAMIC).content
    assert "denied_static" not in prompt.render()
    assert manifest[0]["name"] == "large_hint"
    assert manifest[0]["status"] == "trimmed"
    assert manifest[0]["included"] is True
    assert manifest[1]["name"] == "denied_static"
    assert manifest[1]["status"] == "target_denied"
    assert manifest[1]["included"] is False
    assert manifest[2]["name"] == "overflow"
    assert manifest[2]["status"] in {"trimmed", "total_budget_exceeded"}
    assert prompt.manifest()["metadata"]["context_injection_policy"][
        "schema_version"
    ] == "agent-core-context-injection-policy/v1"


def test_context_material_selector_builds_prompt_injections_by_semantic_priority() -> None:
    result = DefaultContextMaterialSelector().select(
        ContextMaterialSelectionRequest(
            task="inspect payment auth failure and csrf risk",
            materials=(
                ContextMaterial(
                    name="old_note",
                    content="legacy dns observation",
                    role="memory",
                    priority=1,
                    metadata={"source": "memory"},
                ),
                ContextMaterial(
                    name="auth_trace",
                    content="payment auth callback failed csrf token validation",
                    role="timeline",
                    priority=3,
                    metadata={"source": "trace"},
                ),
                ContextMaterial(
                    name="schema",
                    content='{"required":["risk"]}',
                    role="schema",
                    priority=2,
                    metadata={"source": "runtime"},
                ),
            ),
            max_materials=2,
            max_bytes=512,
        )
    )
    prompt = AgentPromptBuilder().build(
        AgentContextPack(dynamic_task="inspect", injections=result.injections)
    )
    manifest = result.manifest()

    assert manifest["schema_version"] == "agent-core-context-material-selection-result/v1"
    assert manifest["selected_count"] == 2
    assert manifest["dropped_count"] == 1
    assert manifest["statuses"] == {"count_exceeded": 1, "selected": 2}
    assert result.injections[0].name == "auth_trace"
    assert result.injections[0].target == PromptBucketRole.TIMELINE_OPEN
    assert result.injections[0].metadata["context_material_selection"]["rank"] == 1
    assert "[context_injection:auth_trace source=trace]" in prompt.bucket(
        PromptBucketRole.TIMELINE_OPEN
    ).content
    assert "[context_injection:schema source=runtime]" in prompt.bucket(
        PromptBucketRole.SEMI_DYNAMIC_2
    ).content
    assert "old_note" not in prompt.render()


def test_context_material_selector_reports_target_score_and_budget_drops() -> None:
    result = DefaultContextMaterialSelector().select(
        ContextMaterialSelectionRequest(
            task="payment auth",
            materials=(
                ContextMaterial(name="empty", content="", role="memory"),
                ContextMaterial(name="denied", content="must not enter static", role="system"),
                ContextMaterial(name="low", content="unrelated note", role="memory"),
                ContextMaterial(
                    name="huge",
                    content="payment auth " + ("x" * 200),
                    role="timeline",
                    priority=4,
                ),
                ContextMaterial(
                    name="fit",
                    content="payment auth retry",
                    role="memory",
                    priority=3,
                ),
            ),
            allowed_targets=(PromptBucketRole.SEMI_DYNAMIC_1, PromptBucketRole.TIMELINE_OPEN),
            min_score=1.0,
            max_bytes=80,
        )
    )
    by_name = {selection.material.name: selection for selection in result.selections}
    manifest = result.manifest()

    assert by_name["empty"].status == "empty"
    assert by_name["denied"].status == "target_denied"
    assert by_name["low"].status == "score_below_threshold"
    assert by_name["huge"].status == "budget_exceeded"
    assert by_name["fit"].status == "selected"
    assert result.injections[0].name == "fit"
    assert manifest["statuses"] == {
        "budget_exceeded": 1,
        "empty": 1,
        "score_below_threshold": 1,
        "selected": 1,
        "target_denied": 1,
    }
    assert manifest["targets"] == {"timeline_open": 1}


def test_context_material_selector_places_stable_skill_and_volatile_memory_materials() -> None:
    result = DefaultContextMaterialSelector().select(
        ContextMaterialSelectionRequest(
            task="payment auth",
            materials=(
                ContextMaterial(
                    name="auth_skill",
                    content="payment auth validation checklist",
                    role="skill",
                    priority=4,
                ),
                ContextMaterial(
                    name="auth_memory",
                    content="payment auth failed before on retry",
                    role="memory",
                    priority=3,
                ),
            ),
            max_materials=2,
            max_bytes=256,
        )
    )

    by_name = {selection.material.name: selection for selection in result.selections}

    assert by_name["auth_skill"].target == PromptBucketRole.SEMI_DYNAMIC_1
    assert by_name["auth_memory"].target == PromptBucketRole.TIMELINE_OPEN
    assert {
        injection.name: injection.target for injection in result.injections
    } == {
        "auth_skill": PromptBucketRole.SEMI_DYNAMIC_1,
        "auth_memory": PromptBucketRole.TIMELINE_OPEN,
    }


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

