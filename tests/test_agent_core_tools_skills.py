from __future__ import annotations

import zipfile

import pytest

from agent_core.actions import ActionRegistry
from agent_core.prompt import PromptAssembler, PromptBucketRole
from agent_core.react import ReActConfig, ReActExecutor
from agent_core.skills import SkillRegistry, SkillSpec
from agent_core.testing import InMemoryHarness, MockLLMProvider
from agent_core.tools import (
    ToolExecutionCenter,
    ToolInvocation,
    ToolRegistry,
    ToolResult,
    ToolRetryPolicy,
    ToolSpec,
)


class _FlakyToolRuntime:
    def __init__(self, results: tuple[ToolResult | Exception, ...]) -> None:
        self.results = list(results)
        self.invocations: list[ToolInvocation] = []

    def specs(self) -> tuple[ToolSpec, ...]:
        return (ToolSpec(name="flaky", description="Flaky tool"),)

    async def invoke(self, invocation: ToolInvocation) -> ToolResult:
        self.invocations.append(invocation)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.mark.asyncio
async def test_tool_registry_registers_validates_and_invokes_tools() -> None:
    registry = ToolRegistry()

    async def handler(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            content=f"hello {invocation.arguments['name']}",
        )

    registry.register(
        ToolSpec(
            name="hello",
            description="Say hello",
            parameters_schema={
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
            },
        ),
        handler,
    )

    result = await registry.invoke(ToolInvocation(tool_name="hello", arguments={"name": "yak"}))
    bad = await registry.invoke(ToolInvocation(tool_name="hello", arguments={"name": 123}))

    assert result.ok
    assert result.content == "hello yak"
    assert bad.status == "failed"
    assert "must be string" in bad.error
    assert "hello: Say hello" in registry.render_inventory()


@pytest.mark.asyncio
async def test_tool_execution_center_retries_retryable_results_and_records_attempts() -> None:
    first = ToolResult(
        call_id="call-1",
        tool_name="flaky",
        status="failed",
        error="temporary",
        metadata={"retryable": True},
    )
    second = ToolResult(call_id="call-1", tool_name="flaky", content="ok")
    runtime = _FlakyToolRuntime((first, second))
    center = ToolExecutionCenter(runtime, retry_policy=ToolRetryPolicy(max_attempts=2))

    result = await center.invoke(ToolInvocation(tool_name="flaky", call_id="call-1"))
    manifest = center.manifest()

    assert result.ok
    assert result.content == "ok"
    assert result.metadata["tool_execution"]["attempt_count"] == 2
    assert result.metadata["tool_execution"]["retried"] is True
    assert manifest["schema_version"] == "agent-core-tool-execution-center/v1"
    assert manifest["records"][0]["attempts"][0]["retryable"] is True
    assert manifest["records"][0]["attempts"][1]["status"] == "completed"


@pytest.mark.asyncio
async def test_tool_execution_center_converts_retryable_exceptions_to_final_results() -> None:
    first = RuntimeError("network hiccup")
    setattr(first, "retryable", True)
    runtime = _FlakyToolRuntime((first,))
    center = ToolExecutionCenter(runtime, retry_policy=ToolRetryPolicy(max_attempts=1))

    result = await center.invoke(ToolInvocation(tool_name="flaky", call_id="call-1"))

    assert result.status == "failed"
    assert result.error == "network hiccup"
    assert result.metadata["exception_type"] == "RuntimeError"
    assert result.metadata["tool_execution"]["attempt_count"] == 1


@pytest.mark.asyncio
async def test_tool_registry_searches_and_budgets_inventory() -> None:
    registry = ToolRegistry()

    async def handler(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(call_id=invocation.call_id, tool_name=invocation.tool_name)

    for name, desc, tags in [
        ("http_probe", "Probe HTTP service", ("web", "recon")),
        ("dns_lookup", "Resolve DNS names", ("dns", "recon")),
        ("code_scan", "Scan source code", ("code",)),
    ]:
        registry.register(ToolSpec(name=name, description=desc, tags=tags), handler)

    search = registry.search("http recon")
    selection = registry.select_inventory(max_tokens=8)
    rendered = registry.render_inventory(max_tokens=8)

    assert search[0].name == "http_probe"
    assert selection.visible_tools
    assert selection.omitted_count > 0
    assert "Use search_tools" in rendered


@pytest.mark.asyncio
async def test_tool_registry_search_prefers_name_and_relevance() -> None:
    registry = ToolRegistry()

    async def handler(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(call_id=invocation.call_id, tool_name=invocation.tool_name)

    registry.register(ToolSpec(name="http_probe", description="Probe web service"), handler)
    registry.register(ToolSpec(name="probe_notes", description="Unrelated notes about probing"), handler)

    results = registry.search("http probe")

    assert [tool.name for tool in results][:2] == ["http_probe", "probe_notes"]


@pytest.mark.asyncio
async def test_tool_registry_registers_functions_from_signature() -> None:
    registry = ToolRegistry()

    @registry.register_function(description="Add two numbers", aliases=("sum",), tags=("math",))
    def add(a: int, b: int) -> int:
        return a + b

    spec = registry.get("sum")
    result = await registry.invoke(ToolInvocation(tool_name="sum", arguments={"a": 2, "b": 3}))
    bad = await registry.invoke(ToolInvocation(tool_name="add", arguments={"a": "x", "b": 3}))

    assert spec is not None
    assert spec.parameters_schema["required"] == ["a", "b"]
    assert spec.parameters_schema["properties"]["a"]["type"] == "integer"
    assert result.content == "5"
    assert bad.status == "failed"


@pytest.mark.asyncio
async def test_react_executor_can_use_tool_registry_runtime() -> None:
    registry = ToolRegistry()

    async def handler(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            content="registered tool result",
        )

    registry.register(ToolSpec(name="lookup", description="Lookup target"), handler)
    provider = MockLLMProvider(
        [
            {"action": "call_tool", "arguments": {"tool_name": "lookup", "arguments": {}}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=registry,
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        config=ReActConfig(max_iterations=3),
    )

    result = await executor.run("use registered tool", PromptAssembler().build())

    assert result.status == "completed"
    assert provider.requests[1].messages[-1].content == "registered tool result"


def test_skill_registry_selects_and_renders_prompt_material() -> None:
    registry = SkillRegistry()
    registry.register(
        SkillSpec(
            name="code-review",
            description="Review code",
            prompt="Check correctness before style.",
            tags=("code", "review"),
            priority=10,
        )
    )
    registry.register(
        SkillSpec(
            name="web-recon",
            description="Collect web evidence",
            prompt="Prefer passive discovery first.",
            tags=("recon", "web"),
            priority=5,
        )
    )

    selected = registry.select("please review this code", limit=2)
    rendered = registry.render_prompt(selected)

    assert [skill.name for skill in selected] == ["code-review"]
    assert "[skill:code-review]" in rendered
    assert "Check correctness before style." in rendered


def test_skill_registry_respects_disable_model_invocation_for_auto_select() -> None:
    registry = SkillRegistry()
    registry.register(
        SkillSpec(
            name="manual-only",
            description="Manual code helper",
            prompt="manual",
            tags=("code",),
            priority=100,
            disable_model_invocation=True,
        )
    )
    registry.register(
        SkillSpec(
            name="auto-code",
            description="Automatic code helper",
            prompt="auto",
            tags=("code",),
            priority=1,
        )
    )

    selected = registry.select("code task")
    explicitly_tagged = registry.select("other task", tags=("code",))

    assert [skill.name for skill in selected] == ["auto-code"]
    assert [skill.name for skill in explicitly_tagged] == ["manual-only", "auto-code"]


def test_skill_registry_discovers_markdown_skills_and_loads_resources(tmp_path) -> None:
    skill_dir = tmp_path / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: review
description: Review code
tags: [code, review]
---
# Review

<!-- include: references/rules.md -->
""",
        encoding="utf-8",
    )
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "rules.md").write_text("line one\nline two\nline three", encoding="utf-8")

    registry = SkillRegistry()
    discovered = registry.discover_markdown_skills(tmp_path / "skills")
    window = registry.load_resource("@review/references/rules.md", offset=2, max_bytes=512)
    rendered, truncated = window.render()

    assert [skill.name for skill in discovered] == ["review"]
    assert registry.get("review").metadata["content_hash"]  # type: ignore[union-attr]
    assert window.skill_name == "review"
    assert "2 | line two" in rendered or "line two" in rendered
    assert truncated is False


def test_skill_registry_discovers_zip_archive_skills_and_loads_resources(tmp_path) -> None:
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "bundle/review/SKILL.md",
            """---
name: archived-review
description: Review from archive
tags: [code]
---
# Archived Review

<!-- include: references/rules.md -->
""",
        )
        bundle.writestr("bundle/review/references/rules.md", "first\nsecond\nthird")

    registry = SkillRegistry()
    discovered = registry.discover_archive_skills(archive, extract_root=tmp_path / "extracted")
    window = registry.load_resource("@archived-review/references/rules.md", offset=2)
    rendered, truncated = window.render()
    registered = registry.get("archived-review")

    assert [skill.name for skill in discovered] == ["archived-review"]
    assert registered is not None
    assert registered.metadata["archive_path"] == str(archive.resolve())
    assert "2 | second" in rendered or "second" in rendered
    assert truncated is False


def test_skill_registry_rejects_unsafe_zip_archive_members(tmp_path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../SKILL.md", "# Escape")

    registry = SkillRegistry()

    with pytest.raises(ValueError, match="escapes destination"):
        registry.discover_archive_skills(archive, extract_root=tmp_path / "extracted")


def test_skill_markdown_frontmatter_keeps_extended_metadata() -> None:
    from agent_core.skills import skill_from_markdown

    skill = skill_from_markdown(
        """---
name: deploy
description: Deploy app
license: MIT
compatibility: linux
disable-model-invocation: true
metadata:
  author: tester
tags: [deploy, ops]
---
# Deploy

Run deployment checks.
"""
    )

    assert skill.license == "MIT"
    assert skill.compatibility == "linux"
    assert skill.disable_model_invocation is True
    assert skill.metadata["metadata"]["author"] == "tester"
    assert skill.manifest()["license"] == "MIT"


def test_skills_can_feed_prompt_semi_dynamic_bucket() -> None:
    registry = SkillRegistry()
    registry.register(
        SkillSpec(
            name="timeline-hygiene",
            prompt="Keep recent observations separate from compressed history.",
            tags=("timeline",),
            priority=1,
        )
    )
    selected = registry.select("timeline reducer task")
    prompt = (
        PromptAssembler()
        .add(PromptBucketRole.HIGH_STATIC, "system")
        .add(PromptBucketRole.SEMI_DYNAMIC_1, registry.render_prompt(selected))
        .build()
    )

    assert "timeline-hygiene" in prompt.bucket(PromptBucketRole.SEMI_DYNAMIC_1).content

