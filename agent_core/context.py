"""Context material types and assembly protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_core.capabilities import CapabilityCatalog
from agent_core.prompt import PromptAssembler, PromptBucketRole, PromptIR
from agent_core.skills import SkillsContext
from agent_core.timeline import TimelineBudget, TimelineStore
from agent_core.tools import ToolRuntimePort


@dataclass(frozen=True)
class ContextBudget:
    max_prompt_bytes: int = 128 * 1024
    max_timeline_bytes: int = 64 * 1024
    recent_keep_ratio: float = 0.25


@dataclass(frozen=True)
class ContextMaterial:
    name: str
    content: str
    role: str = "dynamic"
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextObservation:
    prompt_manifest: dict[str, Any]
    notes: tuple[str, ...] = ()


class ContextAssemblerPort(Protocol):
    async def assemble(self, task: str, budget: ContextBudget) -> PromptIR:
        """Build a prompt IR for one agent turn."""


@dataclass(frozen=True)
class AgentContextPack:
    system: str = ""
    task_instruction: str = ""
    schema: str = ""
    output_example: str = ""
    recent_tools_cache: str = ""
    user_history: str = ""
    workspace: str = ""
    current_time: str = ""
    dynamic_task: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentPromptBuilder:
    """Build Yaklang-style six-bucket PromptIR from core registries."""

    def __init__(
        self,
        *,
        tools: ToolRuntimePort | None = None,
        skills: SkillsContext | None = None,
        timeline: TimelineStore | None = None,
        timeline_budget: TimelineBudget | None = None,
        capabilities: CapabilityCatalog | None = None,
    ) -> None:
        self.tools = tools
        self.skills = skills
        self.timeline = timeline
        self.timeline_budget = timeline_budget or TimelineBudget()
        self.capabilities = capabilities

    def build(self, context: AgentContextPack) -> PromptIR:
        assembler = PromptAssembler()
        assembler.add(PromptBucketRole.HIGH_STATIC, context.system)
        assembler.add(PromptBucketRole.FROZEN, self._frozen_materials())
        assembler.add(PromptBucketRole.SEMI_DYNAMIC_1, self._semi_dynamic_1(context))
        assembler.add(PromptBucketRole.SEMI_DYNAMIC_2, self._semi_dynamic_2(context))
        assembler.add(PromptBucketRole.TIMELINE_OPEN, self._timeline_open(context))
        assembler.add(PromptBucketRole.DYNAMIC, context.dynamic_task)
        return assembler.build(metadata=context.metadata)

    def _frozen_materials(self) -> str:
        parts: list[str] = []
        if self.capabilities is not None:
            catalog = self.capabilities.render_prompt(include_skills=False)
            if catalog:
                parts.append("[capability_catalog]\n" + catalog)
        elif self.tools is not None:
            render_inventory = getattr(self.tools, "render_inventory", None)
            if callable(render_inventory):
                inventory = str(render_inventory())
            else:
                inventory = "\n".join(
                    f"- {spec.name}: {spec.description}".rstrip() for spec in self.tools.specs()
                )
            if inventory:
                parts.append("[tool_inventory]\n" + inventory)
        if self.timeline is not None:
            frozen = self.timeline.view(self.timeline_budget).render_frozen()
            if frozen:
                parts.append("[timeline_frozen]\n" + frozen)
        return "\n\n".join(parts)

    def _semi_dynamic_1(self, context: AgentContextPack) -> str:
        parts: list[str] = []
        if self.skills is not None:
            skills_context = self.skills.render_stable()
            if skills_context:
                parts.append("[skills_context]\n" + skills_context)
        if context.recent_tools_cache:
            parts.append("[recent_tools_cache]\n" + context.recent_tools_cache)
        return "\n\n".join(parts)

    @staticmethod
    def _semi_dynamic_2(context: AgentContextPack) -> str:
        parts = []
        if context.task_instruction:
            parts.append("[task_instruction]\n" + context.task_instruction)
        if context.schema:
            parts.append("[schema]\n" + context.schema)
        if context.output_example:
            parts.append("[output_example]\n" + context.output_example)
        return "\n\n".join(parts)

    def _timeline_open(self, context: AgentContextPack) -> str:
        parts: list[str] = []
        if self.timeline is not None:
            open_tail = self.timeline.view(self.timeline_budget).render_open()
            if open_tail:
                parts.append("[timeline_open]\n" + open_tail)
        if context.workspace:
            parts.append("[workspace]\n" + context.workspace)
        if context.user_history:
            parts.append("[user_history]\n" + context.user_history)
        if context.current_time:
            parts.append("[current_time]\n" + context.current_time)
        return "\n\n".join(parts)

