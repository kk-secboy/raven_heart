"""Portable task contract profiles for SDK-owned run entrypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.backends import StorageBackendRequirement
from agent_core.preflight import AgentRunPreflightRequirements


@dataclass(frozen=True)
class AgentTaskContract:
    """One runtime-neutral contract for a class of agent run requests."""

    name: str
    status: str = "stable_contract"
    description: str = ""
    task_kinds: tuple[str, ...] = ()
    max_task_bytes: int | None = None
    required_actions: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    required_skills: tuple[str, ...] = ()
    required_mcp_servers: tuple[str, ...] = ()
    require_memory: bool = False
    storage_backend_requirements: tuple[StorageBackendRequirement, ...] = ()
    context_contracts: tuple[str, ...] = ()
    prompt_contracts: tuple[str, ...] = ()
    runtime_responsibilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def preflight_requirements(
        self,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRunPreflightRequirements:
        """Build SDK preflight requirements implied by this task contract."""

        return AgentRunPreflightRequirements(
            max_task_bytes=self.max_task_bytes,
            required_actions=self.required_actions,
            required_tools=self.required_tools,
            required_skills=self.required_skills,
            required_mcp_servers=self.required_mcp_servers,
            require_memory=self.require_memory,
            storage_backend_requirements=self.storage_backend_requirements,
            metadata={
                "contract_name": self.name,
                "task_kinds": list(self.task_kinds),
                **dict(self.metadata),
                **dict(metadata or {}),
            },
        )

    def manifest(self) -> dict[str, Any]:
        requirements = self.preflight_requirements()
        return {
            "schema_version": "agent-core-task-contract/v1",
            "name": self.name,
            "status": self.status,
            "description": self.description,
            "task_kinds": list(self.task_kinds),
            "preflight_requirements": requirements.manifest(),
            "context_contracts": list(self.context_contracts),
            "prompt_contracts": list(self.prompt_contracts),
            "runtime_responsibilities": list(self.runtime_responsibilities),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreTaskContractProfile:
    """SDK-owned contract profile for run inputs across host runtimes."""

    contracts: tuple[AgentTaskContract, ...]
    supported_runtime_profiles: tuple[str, ...]
    required_core_contracts: tuple[str, ...]
    forbidden_core_dependencies: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def contract(self, name: str) -> AgentTaskContract:
        for contract in self.contracts:
            if contract.name == name:
                return contract
        raise KeyError(name)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-task-contract-profile/v1",
            "contract_count": len(self.contracts),
            "contracts": [contract.manifest() for contract in self.contracts],
            "contract_names": [contract.name for contract in self.contracts],
            "supported_runtime_profiles": list(self.supported_runtime_profiles),
            "required_core_contracts": list(self.required_core_contracts),
            "forbidden_core_dependencies": list(self.forbidden_core_dependencies),
            "metadata": dict(self.metadata),
        }


def agent_core_task_contract_profile(
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreTaskContractProfile:
    """Return portable run-input contracts supported by the SDK base."""

    return AgentCoreTaskContractProfile(
        contracts=(
            AgentTaskContract(
                name="generic_run_request",
                description="Minimal provider-neutral AgentRunRequest contract for any host runtime.",
                task_kinds=("generic", "code", "ops", "security", "research"),
                max_task_bytes=64_000,
                context_contracts=("AgentContextPack", "ContextInjection"),
                prompt_contracts=("PromptIR", "PromptBucketBudgetPolicy"),
                runtime_responsibilities=(
                    "domain prompts",
                    "tenant identity",
                    "product workflow metadata",
                ),
            ),
            AgentTaskContract(
                name="context_aware_task",
                description="Task contract for semantic context selection, injection, and trimming.",
                task_kinds=("code", "ops", "security", "research"),
                max_task_bytes=48_000,
                required_skills=(),
                context_contracts=(
                    "ContextMaterialSelectionRequest",
                    "ContextMaterialCenter",
                    "MCPContextMaterialRequest",
                    "ContextInjectionPolicy",
                    "ContextWindowBuilder",
                ),
                prompt_contracts=(
                    "PromptBucketBudgetPolicy",
                    "PromptSemanticReducerPort",
                    "AgentPromptBudgetPlan",
                ),
                runtime_responsibilities=(
                    "candidate context collection",
                    "domain ranking models",
                    "runtime-provided hints",
                ),
            ),
            AgentTaskContract(
                name="memory_context_task",
                description="Task contract for runs that require memory and runtime-owned context stores.",
                task_kinds=("code", "ops", "security", "automation"),
                max_task_bytes=32_000,
                require_memory=True,
                storage_backend_requirements=(
                    StorageBackendRequirement(
                        role="memory",
                        allowed_kinds=("sqlite", "markdown", "postgres", "vector", "graph", "custom"),
                        require_queryable=True,
                        required_capabilities=("search",),
                        metadata={"contract": "memory_context_task"},
                    ),
                    StorageBackendRequirement(
                        role="context_material",
                        allowed_kinds=("sqlite", "markdown", "postgres", "vector", "graph", "custom"),
                        require_queryable=True,
                        metadata={"contract": "memory_context_task"},
                    ),
                ),
                context_contracts=("MemoryCenter", "ContextMaterialCenter", "ContextInjectionPolicy"),
                prompt_contracts=("PromptIR", "ContextWindowReport"),
                runtime_responsibilities=(
                    "PG/vector/graph adapters",
                    "memory retention policy",
                    "domain-specific recall policy",
                ),
            ),
            AgentTaskContract(
                name="managed_background_task",
                description="Task contract for queued, cancellable, resumable background runs.",
                task_kinds=("ops", "security", "automation"),
                max_task_bytes=32_000,
                storage_backend_requirements=(
                    StorageBackendRequirement(
                        role="run_state",
                        allowed_kinds=("sqlite", "markdown", "postgres", "product", "custom"),
                        require_durable=True,
                        metadata={"contract": "managed_background_task"},
                    ),
                    StorageBackendRequirement(
                        role="event_log",
                        allowed_kinds=("sqlite", "markdown", "postgres", "product", "custom"),
                        require_durable=True,
                        metadata={"contract": "managed_background_task"},
                    ),
                    StorageBackendRequirement(
                        role="run_trace",
                        allowed_kinds=("sqlite", "markdown", "postgres", "object_storage", "custom"),
                        require_durable=True,
                        metadata={"contract": "managed_background_task"},
                    ),
                ),
                context_contracts=("AgentSessionManager", "AgentResumeRequest"),
                prompt_contracts=("TraceReplayHarness", "TraceEvalSpec"),
                runtime_responsibilities=(
                    "worker scheduling",
                    "SSE/WebSocket delivery",
                    "operator dashboards",
                ),
            ),
            AgentTaskContract(
                name="tool_structured_task",
                description="Task contract for tool-heavy runs with structured final output.",
                task_kinds=("code", "ops", "security"),
                max_task_bytes=32_000,
                required_tools=(),
                context_contracts=("ToolCenter", "LLMToolContract", "StructuredOutputSpec"),
                prompt_contracts=("SchemaValidationResult", "TraceEvalSpec"),
                runtime_responsibilities=(
                    "concrete tool implementations",
                    "tool sandboxing",
                    "domain schemas",
                ),
            ),
        ),
        supported_runtime_profiles=("generic", "code", "ops", "security", "research", "automation", "custom"),
        required_core_contracts=(
            "AgentRunRequest",
            "AgentRunPreflightRequirements",
            "AgentRunPreflightCenter",
            "AgentContextPack",
            "ContextMaterialSelectionRequest",
            "ContextInjectionPolicy",
            "PromptBucketBudgetPolicy",
            "PromptSemanticReducerPort",
            "StorageBackendRequirement",
            "AgentSessionManager",
            "TraceEvalSpec",
        ),
        forbidden_core_dependencies=(
            "openai",
            "agents",
            "fastapi",
            "graphiti",
            "sqlalchemy",
            "redis",
        ),
        metadata=dict(metadata or {}),
    )
