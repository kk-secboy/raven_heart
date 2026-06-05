# raven_heart

`raven_heart` is a pure, provider-neutral Harness/ReAct LLM Agent SDK base.
It is designed to be imported by different runtimes such as Raven, code agents,
ops agents, research agents, and future automation systems.

中文一句话：这是一个纯独立 agent SDK 基座，只负责 agent 的通用核心能力，不包含 Raven、OpenAI Agents SDK、Graphiti 或其他具体 runtime adapter。

## Scope

`agent_core` contains only reusable agent mechanics:

- ReAct loop and structured action execution.
- Structured final output contracts with validation and repair feedback.
- Harness lifecycle, checkpoint, resume, trace, and replay primitives.
- LLM provider abstractions and provider routing.
- Provider model capability profiles for context windows, streaming, tool-call, JSON, and structured-output routing.
- Provider call records and usage/failure manifests.
- Tool registry and tool center.
- Tool replay records and replay store port.
- Skill center and skill context injection.
- MCP center and SDK-free stdio connector.
- Prompt buckets and context trimming.
- Provider-neutral prompt IR with semantic bucket trimming.
- Prompt bucket budget policy for per-bucket caps and audit manifests.
- Context reducer port and deterministic timeline reduction.
- Runner-level automatic timeline reduction before prompt assembly.
- Context material store center for SDK/runtime-owned candidate context backends.
- Context injection records for resume, memory, runtime hints, and other bucketed material.
- Context injection policy for bucket allow-lists, per-injection trimming, total injection budget, and audit manifests.
- Context window reports for selected, injected, trimmed, and excluded context material.
- Provider-neutral embedding request/response, provider routing, and semantic ranking contracts.
- SQLite, Markdown, and in-memory stores for lightweight memory.
- Memory governance decisions for allow/rewrite/deny write auditing.
- Unified storage backend manifests for core stores and runtime-owned backends.
- Portable state bundle contracts for migration, archive, and recovery preflight.
- External memory store wrapper for runtime-owned PG/vector/graph/product adapters.
- Pluggable journal stores for harness checkpoint/resume persistence.
- Journal replay manifests and snapshot consistency audit.
- Run trace bundle for provider/tool/approval/journal/event audit aggregation.
- Run trace store port plus in-memory, SQLite, and Markdown trace stores.
- Trace replay, evaluation harness, and multi-case eval suites for provider-neutral regression gates.
- Manager run state store port plus in-memory, SQLite, and Markdown stores.
- Sequenced event stream plus in-memory, SQLite, and Markdown event logs.
- Planner protocol plus in-memory, SQLite, and Markdown plan state stores.
- Multi-agent handoff specs, routing decisions, and lightweight coordination.
- Human-in-loop approval request and decision queue primitives.
- Approval resume context for approved action/tool gate continuation.
- Policy gates, budget metadata, loop guards, and capability manifests.
- Package-level SDK manifest for public API, capability, storage, and runtime-boundary audits.
- Runtime-neutral task contracts for generic, context-aware, memory-backed, managed, and tool/structured runs.
- Structured output specs and validator ports for provider-neutral final answers.

Runtime integration is intentionally outside this repository. Raven, OpenAI Agents SDK, Graphiti, Anthropic, OpenAI, local models, file-system tools, CI runners, and product APIs should connect to `agent_core` from their own runtime packages or repositories.

## Non-Goals

This repository does not contain:

- Raven runtime adapters.
- OpenAI Agents SDK compatibility adapters.
- Graphiti adapters.
- Concrete pentest tools.
- FastAPI, gateway, UI, database, or deployment code.
- Yaklang source translation.

Those can be built later in RavenStorm or separate adapter repositories. Keeping them out makes this package usable by code agents and other non-Raven agents.

## Design Boundary

| Core owns | Runtime owns |
| --- | --- |
| Agent loop semantics | Product orchestration |
| Prompt bucket IR | Domain-specific prompt content |
| Provider protocol, routing, and call audit | Concrete LLM clients, credentials, and rate limits |
| Tool registry protocol | Real tools, sandboxing, permissions |
| Tool replay protocol and manifests | Durable replay backend and retention policy |
| Skill registry | Skill distribution UX |
| MCP center interfaces | MCP server deployment and secrets |
| Memory and journal ports | Graphiti, RAG, durable product stores |
| Trace replay/eval harness and generic eval suite runner | Product datasets, dashboards, and domain scoring policy |
| Manager run state store | Product workflow DB, scheduling, cross-process workers |
| Planner protocol and plan state | Domain-specific planning strategy and product workflow |
| Multi-agent handoff protocol | Product queues, distributed workers, business orchestration |
| Context reducer protocol | Domain summarizers, archive storage, and retrieval strategy |
| Structured output contracts | Domain schemas and downstream business handling |
| Approval request and decision queue | Approval UI, identity, permissions, and workflow routing |
| Harness state and event log contracts | API routes, UI streams, metrics, production persistence backend |

The dependency direction must always be:

```text
runtime imports agent_core
agent_core never imports runtime
```

## SDK Manifest

`agent_core.agent_core_sdk_manifest()` returns a machine-readable package
manifest for migration and runtime preflight checks. It records:

- Public API exported from the package root.
- Core capability matrix for harness, ReAct, providers, tools, skills, MCP,
  memory, prompt/context shaping, policy, trace/replay/eval, and coordination.
- Storage backend interface roles plus built-in `in_memory` / `sqlite` /
  `markdown` kinds and external `postgres` / `vector` / `graph` /
  `object_storage` / `product` / `custom` kinds.
- Runtime boundary rules, including forbidden runtime dependencies and adapter
  packages that must stay outside this repository.

The manifest is intentionally provider-neutral. A RavenStorm runtime, code
agent, ops agent, or separate adapter package can read it to assert that it is
integrating against the SDK base instead of importing Raven/OpenAI Agents
SDK/Graphiti/FastAPI runtime code back into `agent_core`.

`agent_core.evaluate_agent_core_readiness()` checks that manifest against the
default replacement-readiness profile. The report answers a narrower question:
"does this SDK package expose the generic core contracts a host runtime needs
before adapter work starts?" It verifies required capabilities, public API
contracts, storage backend roles/kinds, and runtime-boundary declarations. It
does not validate RavenStorm-specific wiring, credentials, UI behavior, or
production adapters; those remain runtime acceptance tests.

`agent_core.agent_core_api_contract()` and
`agent_core.evaluate_agent_core_api_stability()` define the package-root API
stability gate. The contract separates stable integration entrypoints from MVP
exports that may still change before 1.0. The report blocks if a stable API name
is missing, which lets host runtimes pin migration checks before depending on
`raven_heart` as their agent base.

`agent_core.evaluate_agent_core_api_lifecycle()` adds the API lifecycle gate. It
checks the same package manifest against the stable/MVP/experimental/deprecated
groups, reports the package version and pre-1.0 status, enforces that public API
names are classified by the contract, and keeps deprecation policy
machine-readable for future SDK releases.

`agent_core.evaluate_agent_core_runtime_boundary()` audits the SDK package
itself for forbidden runtime imports and adapter package directories. This is
the self-check version of the boundary rule: it proves `agent_core` stayed pure
before a host runtime starts wiring RavenStorm, OpenAI Agents SDK compatibility,
Graphiti, FastAPI, or vendor clients outside this repository.

`agent_core.run_agent_core_acceptance()` runs a deterministic pure-SDK
acceptance scenario. It exercises readiness, ReAct, provider routing, ToolCenter,
memory recall, context injection, journal persistence, event logging, tool
replay, trace bundling, and trace eval without importing a host runtime. The
result is an `AgentCoreAcceptanceReport` with blocking issues, run summary,
readiness report, and trace-eval report. This is the package-level gate before a
runtime starts adapter-specific migration tests.

`agent_core.run_agent_core_context_acceptance()` runs a deterministic
Yaklang-style context gate. It exercises context material routing, semantic
selection, target denial, context injection, per-injection trimming, bucket-local
prompt budgets, semantic prompt trimming, provider-aware prompt budget, and
trace eval. The report proves the portable context pipeline works before a
runtime adds product-specific stores, ranking models, or domain prompt content.

`agent_core.run_agent_core_approval_acceptance()` runs deterministic
policy/approval checks. It verifies that a policy gate creates a pending
approval record, an approved decision becomes `ApprovalResumeContext`, resumed
approval material is injected into the prompt, the gated tool executes after
approval, and approval/policy events are trace-evaluable. Operator UX, identity,
notifications, ticketing, and workflow routing remain runtime responsibilities.

`agent_core.run_agent_core_orchestration_acceptance()` runs deterministic
capability orchestration checks. It verifies unified discovery across actions,
local tools, loaded skills, MCP tools, MCP resources, MCP prompts, and MCP
servers; ToolCenter routing/execution for local and MCP-backed tools; MCP
inventory refresh; MCP context-material export; capability prompt rendering;
and ToolCenter/MCP/skill trace summaries. Concrete tools, MCP processes,
credentials, network sessions, and product workflow queues remain outside the
SDK.

`agent_core.run_agent_core_capability_governance_acceptance()` runs deterministic
capability governance checks. It verifies disabled local/MCP tools are hidden
from default model-visible inventories while still appearing in route-plan audit
candidates, disabled tool invocations fail closed, failed/disabled MCP servers
are isolated and traced, and skills disabled for model invocation are not
auto-selected. Operator identity, product permissions, distributed MCP
supervision, and approval UX remain runtime responsibilities.

`agent_core.run_agent_core_coordination_acceptance()` runs deterministic
planner/handoff/agent-as-tool/artifact checks. It executes a two-step plan
through `PlanExecutor`, routes a handoff to a capable reviewer session, exposes
that reviewer as an agent tool, stores coordination evidence as an artifact, and
evaluates planner, handoff, agent-tool, and artifact traces. Distributed
scheduling, product workflow queues, and runtime-specific worker policy remain
outside the SDK.

`agent_core.run_agent_core_concurrency_acceptance()` runs deterministic
concurrent session-isolation checks. It starts two managed sessions at the same
time, proves the manager observes both active before release, keeps provider
requests and tool invocations isolated per session, tails terminal event streams
by run key/session name, stores both traces in one shared trace store, and
passes trace eval for each run. Distributed locks, worker pools, process
supervision, and UI scheduling remain runtime responsibilities.

`agent_core.run_agent_core_event_acceptance()` runs deterministic streaming and
event-stream checks. It verifies streaming provider runs emit prompt-safe
`model_stream` events, event logs can be paged and tailed by cursor, managed
background runs expose events before and after completion, and trace eval can
require event-log types plus monotonic sequencing. SSE, WebSocket, metrics,
dashboards, and production observability sinks remain runtime responsibilities.

`agent_core.run_agent_core_external_backend_acceptance()` runs deterministic
backend portability checks. It verifies built-in SQLite and Markdown backend
manifests, runtime-owned postgres/vector/graph/product backend contracts,
external memory and context-material calls, storage preflight selection,
context injection, selected context material, and trace eval. Concrete PG,
vector DB, graph/Graphiti, product API, migration, tenancy, and retention
drivers remain runtime responsibilities.

`agent_core.run_agent_core_guardrail_acceptance()` runs deterministic guardrail
checks. It verifies input preflight blocking, policy denial for dangerous tools
without executing them, structured-output validation and repair, policy decision
audit records, failure summaries, and trace eval. Product-specific safety
policies, operator escalation, and domain eval suites remain runtime
responsibilities.

`agent_core.run_agent_core_interaction_acceptance()` runs deterministic mixed
interaction checks. It verifies one `AgentRunner` task can use streaming provider
calls, receive a streamed provider-native tool call, execute the tool through the
SDK runtime, send the tool result back as a provider-neutral tool message, finish
with structured JSON output, emit event-stream records, and pass trace eval for
the combined stream/tool/structured contract. Concrete provider credentials,
runtime SSE/WebSocket delivery, and UI presentation remain outside the SDK.

`agent_core.run_agent_core_lifecycle_acceptance()` runs deterministic
scheduling/lifecycle checks. It verifies single-process capacity queueing and
dequeue, queued-run cancellation before capacity claim, active-run cancellation,
request timeouts, interrupt metadata, schedule snapshots, event logs, and trace
eval. Distributed schedulers, leases, worker process control, and UI controls
remain runtime responsibilities.

`agent_core.run_agent_core_interrupt_acceptance()` runs deterministic
interrupt/deadline checks. It verifies pre-cancelled runs do not consume
provider capacity, provider and tool deadlines finish as terminal timeout
states with checkpoint phases, manager-driven active cancellation records
interrupt metadata, and the same session can run successfully after cancellation
without a poisoned cancel token. Operator UX, distributed process cancellation,
and product workflow controls remain runtime responsibilities.

`agent_core.run_agent_core_native_tool_acceptance()` runs deterministic
provider-native tool-call checks. It verifies that `AgentRunner` sends native
tool contracts, receives provider tool calls, executes the referenced tool
through the SDK tool runtime, returns a provider-neutral `role=tool` message,
preserves `provider_tool_call` metadata in the journal, and passes trace eval
for provider tool calls/results. Provider-specific HTTP payloads, credentials,
deployment names, and SDK clients remain runtime/provider-adapter
responsibilities.

`agent_core.run_agent_core_provider_acceptance()` runs a deterministic provider
compatibility matrix. It checks provider routing for text, multimodal,
structured-output, and native-tool requests; OpenAI-compatible codec
encode/decode; the default provider-neutral transport codec; prompt-safe request
manifests; streamed tool-call and retryable-error decoding; transport-backed
provider fallback; streaming route selection; and prompt-safe call manifests.
Concrete HTTP clients, credentials, rate limits, and vendor-specific adapters
remain outside the SDK.

`agent_core.agent_core_provider_contract_profile()` declares the SDK-supported
provider adapter API shapes: direct `LLMProviderPort`, SDK transport plus codec,
OpenAI Chat Completions-compatible codec, and custom codec transport. The
profile also lists the core contracts each adapter depends on and the
runtime-owned pieces that must stay outside `agent_core`, including HTTP
clients, credentials, endpoint routing, tenant policy, and vendor SDKs.

`agent_core.run_agent_core_provider_conformance()` runs provider-neutral
conformance checks against one `LLMProviderPort`. With no provider argument it
uses a deterministic no-network provider for SDK validation; runtimes can pass a
real provider adapter when credentials are available and check text completion,
streaming, JSON mode, and native tool-call behavior through the same report
shape. HTTP clients, API keys, deployment routing, and rate-limit handling stay
outside `agent_core`.

`agent_core.run_agent_core_provider_resilience_acceptance()` runs deterministic
provider retry and fallback checks. It verifies retryable completion failures
retry before fallback, retry-after policy can suppress retries, non-retryable
failures move to fallback without retrying, explicit provider requests fail
closed instead of silently falling back, streaming errors can retry and recover
through a fallback stream, and trace eval records provider error classification
for each scenario. Concrete provider credentials, runtime circuit breakers,
tenant retry budgets, and vendor failover policy remain outside the SDK.

`agent_core.run_agent_core_budget_acceptance()` runs deterministic provider
budget checks for `LLMProviderCenter`. It verifies estimated-cost preflight,
actual-cost enforcement, request-level budget overrides, call-attempt limits,
input/total token limits, usage accounting, and trace eval for provider cost.
Vendor quotas, tenant rate limits, billing APIs, and operator policy stay
outside `agent_core`.

`agent_core.run_agent_core_redaction_acceptance()` runs deterministic prompt-safe
redaction checks. It verifies `RedactionPolicy` recursively redacts sensitive
keys, bearer/API-key/password-like values, inline secret strings, and oversized
strings while preserving digest, byte-count, and path audit decisions. Secret
stores, tenant-specific redaction policy, incident workflow, and credential
rotation stay outside `agent_core`.

`agent_core.run_agent_core_eval_suite_acceptance()` runs deterministic multi-case
trace eval suite checks. It proves `TraceEvalSuiteRunner` can bind different
`TraceEvalCase` objects to different run IDs and `TraceEvalSpec` contracts,
aggregate pass/fail status, and report missing traces as blocked cases instead
of raising host-runtime errors. Domain datasets, regression dashboards, and
business-specific scoring stay outside `agent_core`.

`agent_core.run_agent_core_trace_replay_acceptance()` runs deterministic trace
replay compatibility checks. It builds a current SDK run trace, evaluates the
stored trace through `TraceEvalHarness`, verifies a normalized compatible trace
copy has the same replay shape, proves the comparator detects a missing provider
replay step, and confirms a legacy minimal trace can still replay and eval.
Runtime retention, dashboarding, and historical backfill jobs stay outside the
SDK.

`agent_core.run_agent_core_storage_acceptance()` runs deterministic storage
backend portability checks. It verifies built-in in-memory/SQLite/Markdown/none
backend manifests across SDK store roles, SQLite and Markdown memory/context
roundtrips, backend manifest extraction from components, runtime-owned
Postgres/vector/object-storage/product backend contracts, positive multi-role
preflight selection, and a negative blocked preflight. Concrete PG/vector/graph
drivers, object-store clients, product databases, migrations, tenancy, and
retention policies remain runtime responsibilities.

`agent_core.run_agent_core_recovery_acceptance()` runs deterministic recovery
checks for provider fallback and tool retry. The report proves that retryable
provider failures are classified, retried, and recovered through fallback, and
that retryable tool failures record attempts and finish successfully. Host
runtimes still own production retry budgets, rate-limit policy, circuit
breakers, and provider-specific failover rules.

`agent_core.run_agent_core_durable_session_acceptance()` runs deterministic
durable-session checks. It executes the same `AgentRunner` session through
`AgentSessionManager` twice, once with SQLite stores and once with Markdown
stores, then reopens every store path and verifies journal/checkpoint/trace,
event log, tool replay, memory, context material, policy decisions, and manager
run state can be queried again. Production PG/vector/graph/object-store/product
stores remain runtime-owned implementations of the same ports.

`agent_core.run_agent_core_resume_acceptance()` runs deterministic
checkpoint/resume checks. It creates a checkpoint, selects a resume plan,
continues through `AgentRunner.resume()`, injects resumed state into the prompt,
evaluates the run trace, and verifies strict terminal checkpoints are rejected
when `allow_terminal=False`. Runtime worker scheduling and user-facing recovery
flows remain outside the SDK.

`agent_core.run_agent_core_packaging_acceptance()` runs deterministic package
readiness checks. It verifies `pyproject.toml` metadata, explicit build-system
configuration, zero runtime dependencies, package discovery for `agent_core`,
stable public imports, required repository files, and example entrypoint smoke
structure. Release automation, publishing credentials, signing, and adapter
wheels remain outside the SDK.

`agent_core.run_agent_core_task_profile_acceptance()` runs deterministic
cross-runtime task-profile checks. It runs code, ops, and security task profiles
through the same `AgentRunner`, provider center, ToolCenter, SkillsContext,
memory recall, context-material selection, event log, and trace-eval pipeline.
Concrete code tools, ops integrations, security tooling, domain prompts, and
product workflows remain runtime responsibilities.

`agent_core.agent_core_task_contract_profile()` declares the SDK-owned task
input contracts a runtime can use before calling `AgentRunner`: generic run
requests, semantic/context-aware tasks, memory-backed tasks, managed background
runs, and tool/structured-output tasks. `AgentRunRequest(task_contract=...)`
automatically converts the selected contract into run preflight requirements
unless explicit `preflight_requirements` are supplied. This keeps task sizing,
required tools/skills/MCP servers, memory requirements, and SQLite/Markdown/PG/
vector/graph storage expectations in the SDK while concrete adapters and domain
workflows stay outside `agent_core`.

`agent_core.run_agent_core_validation()` runs the aggregate SDK gate. It executes
runtime-boundary audit, readiness, API stability, replacement acceptance, context
acceptance, approval acceptance, budget acceptance, redaction acceptance, orchestration acceptance, coordination
acceptance, durable-session acceptance, event acceptance, external-backend
acceptance, guardrail acceptance, lifecycle acceptance, native-tool acceptance,
packaging acceptance, provider acceptance, provider-conformance checks,
eval-suite checks, storage acceptance, task-profile acceptance, recovery acceptance, and resume acceptance,
then returns one
`AgentCoreValidationReport`. This is the default package-level check a host
runtime should pass before starting adapter-specific migration tests.

## Core Capabilities

### Harness

- Run and turn lifecycle.
- In-memory journal.
- Persistent journal backed by `AgentJournalStorePort`.
- Built-in in-memory, SQLite, and Markdown journal stores.
- Postgres or other durable stores can implement the same port outside core.
- Checkpoints and resume tokens.
- `ResumeCandidate` and `ResumeIndex` for manifest-friendly resumable run discovery.
- `ResumePlan` and `ResumePlanIssue` for preflight resume status, selected
  checkpoint, terminal-run warnings, and missing-candidate errors.
- `AgentResumeRequest.allow_terminal` lets runtimes reject terminal-run
  checkpoints during preflight while preserving permissive SDK defaults.
- `AgentRunRequest.resume_token` for injecting checkpoint state into the next run.
- `AgentRunner.resume()`, `AgentSessionManager.resume()`, and
  `AgentSessionManager.start_resume()` for SDK-level resume entrypoints.
- Run manifest and error recording.
- Terminal status checks.
- `RunInterrupt` and `CancelToken` manifests for unified cancel, timeout,
  deadline, and external-interrupt metadata.
- Policy terminal statuses: `denied` and `approval_required`.
- `AgentJournalReplay` for replayable event timelines and journal consistency audit.
- `AgentRunTraceBundle` for per-run prompt, provider, approval, replay,
  journal, and event manifests.
- `TraceExportBuilder` / `export_trace_bundle` for prompt-safe trace/eval
  export bundles with redaction decisions, byte counts, and payload digests.
- `AgentRunStorePort` for manager-level queued/running/completed state.
- `InMemoryAgentRunStore`, `SQLiteAgentRunStore`, and `MarkdownAgentRunStore`
  for lightweight run state persistence.
- `AgentRunQuery` gives run stores and managers the same portable filter
  surface for run keys, sessions, statuses, metadata equality, limits, and
  reverse ordering.
- `AgentManagerConcurrencyPolicy` for single-process active-run capacity limits.
- `AgentManagerConcurrencyPolicy.reject_when_full=False` for SDK-managed
  pending queues when local capacity is full.
- Unclaimed queued runs can be cancelled without consuming later capacity.
- Restored queued/running/cancelling runs are marked `interrupted` with
  `restored_from_status` metadata instead of being auto-executed.
- `AgentManagerScheduleSnapshot` and `AgentManagerCapacityStatus` for
  provider-neutral queue/schedule/capacity audit before runtime scheduling.
- `AgentLifecycleHookCenter` and `AgentLifecycleEvent` for SDK-level
  run-start/run-complete/run-fail hooks with prompt-safe audit records.
- Lifecycle hooks default to non-fatal recording; `fail_fast=True` lets a
  runtime make hook failures block execution.
- `AgentRunPreflightCenter` runs SDK/runtime preflight checks after optional
  refresh and before resume, memory recall, prompt build, or provider calls.
- `AgentRunPreflightRequirements` lets one request require actions, tools,
  skills, MCP servers, memory availability, task byte limits, or storage
  backend readiness.
- Provider route plans are evaluated before prompt build, so a run that
  requires unsupported streaming, structured output, JSON mode, tool calls, or
  an unavailable provider/model is denied before any provider call.
- Storage backend requirements are evaluated from session component manifests
  before prompt build or provider calls, so PG/vector/graph readiness can block
  a run without importing runtime adapters.
- Blocked preflight reports return a denied `AgentRunOutcome` with traceable
  issue codes and without entering the provider/tool loop.

### ReAct

- Provider-neutral ReAct loop.
- Structured action parsing.
- Structured final output validation and repair feedback.
- Built-in actions: `finish`, `fail`, `call_tool`, `search_skill`, `load_skill`, memory actions.
- Loop guard.
- Tool replay.
- Prompt timeline.
- Memory injection.
- Artifact compaction.
- Request-level timeout handling through `AgentRunRequest.timeout_seconds`.
- Timeout checkpoints, `run_timeout` events, and traceable interrupt manifests
  for provider/tool await boundaries.
- Provider-native tool calling can be enabled per session or per request through
  `AgentSession.native_tool_calls` and `AgentRunRequest.native_tool_calls`.

### Multi-Agent Handoff

- `HandoffSpec` advertises a session's handoff capabilities.
- `HandoffRequest` captures task, source, target, and required capabilities.
- `HandoffRouter` selects a target session deterministically by tags, tools,
  skills, priority, and explicit target.
- `MultiAgentCoordinator` runs the selected `AgentSessionManager` session and
  records a `HandoffRecord`.
- `handoff_spec_from_session()` builds a handoff spec from an `AgentSession`.
- `AgentToolRuntime` exposes managed agent sessions as `ToolRuntimePort`
  tools, so a parent ReAct agent can delegate through the normal tool loop.
- `AgentToolTrace` summarizes those agent-as-tool calls by tool, target
  session, child run id, status, iteration count, and prompt-safe byte counts.

The SDK owns handoff contracts, routing decisions, and manifests. Runtimes own
workflow queues, distributed workers, UI orchestration, retry policy, and
domain-specific delegation strategy.

### Structured Output

- `SchemaValidationIssue`, `SchemaValidationResult`, and
  `validate_json_schema_subset()` define a provider-neutral schema validation
  contract for structured output, action arguments, and tool-call parameters.
- `StructuredOutputSpec` defines a request-level final answer contract.
- `JsonStructuredOutputValidator` validates a deterministic JSON schema subset.
- `StructuredOutputValidatorPort` lets runtimes replace validation without
  changing ReAct execution.
- `AgentRunRequest.structured_output` injects the schema into prompt context.
- `ReActExecutor` validates `finish.output`; if invalid, it feeds a repair
  message back into the loop before failing the run.
- `ActionRegistry.validate_result()` exposes the same schema validation result
  shape for parsed actions while preserving the existing throwing `validate()`
  API.

The SDK owns schema injection, validation lifecycle, repair feedback, and
manifests. Runtimes own domain schemas, typed business objects, UI rendering,
and downstream persistence.

### Events

- `AgentEvent` and `EventSinkPort`.
- Monotonic event sequencing inside `ReActExecutor`.
- `EventLogPort` for event logs that can return stored records and manifests.
- `EventStreamCursor` and `EventStreamBatch` for provider-neutral event paging
  that runtime-owned SSE/WebSocket/polling adapters can consume.
- `EventStreamTail` for bounded polling drains across multiple event batches,
  with terminal-run detection and a ready next cursor.
- `EventStreamCursor.run_key` and `session_name` filters for manager-owned
  background runs before the inner ReAct `run_id` is known.
- `AgentSessionManager.event_batch()` for polling one managed run's SDK event
  log by `run_key`.
- `AgentSessionManager.event_tail()` for draining available pages for one
  managed run without owning the transport loop.
- `ListEventSink` for lightweight event capture and manifest export.
- `SQLiteEventSink` and `MarkdownEventSink` for durable local or inspectable
  SDK event logs.
- `approval_requested` events when policy requires human approval.

The SDK owns event schemas, sequencing inside an executor, and lightweight
event-log persistence. Runtimes own WebSocket/SSE rendering, metrics export,
multi-tenant audit storage, retention policy, and observability pipelines.

### Trace Bundle

- `AgentRunTraceBundle` aggregates one run's provider-neutral audit materials.
- `AgentRunOutcome.trace_manifest` includes session, prompt, journal replay,
  provider call audit, tool replay, policy decisions, approvals, event log,
  resume, timeline reduction, capability discovery, memory recall/search, and
  prompt trim manifests when available.
- Summary counters include capability discovery matches, memory hits, and
  storage backend, context injection, memory governance, ToolCenter, MCP center,
  skill center, approval status, and artifact counts, plus whether the prompt
  was semantically trimmed or bucket-budgeted.
- `StorageBackendTrace` collects backend manifests from session, memory,
  replay, approval, policy, event, artifact, and trace components into one
  run-level backend inventory.
- `ContextInjectionTrace` summarizes prompt injection decisions by name,
  source, target bucket, status, included count, excluded count, and trimmed
  count.
- `ContextMaterialSelectionTrace` summarizes candidate context selection by
  selected/dropped status, target bucket, score, byte count, and prompt-safe
  hashes before injection policy is applied.
- `MemoryGovernanceTrace` summarizes memory write allow/rewrite/deny decisions,
  risk levels, stores, and prompt-safe leak hashes.
- `ApprovalTrace` summarizes human approval status, subject, subject kind, and
  decision metadata without depending on an operator UI or workflow engine.
- `ArtifactTrace` summarizes prompt-safe artifact ids, URIs, hashes, sizes,
  content types, kinds, and tool ownership without embedding artifact content.
- `StructuredOutputTrace` summarizes final-output validation attempts,
  successful schemas, repair requests, and validation failures from journal
  checkpoints.
- `TraceExportBundle` turns run trace and eval manifests into prompt-safe
  export records. The SDK owns redaction, hashes, and bundle schema; runtimes
  own export sinks, operator permissions, retention, dashboards, and incident
  workflows.
- `MCPCenterTrace` summarizes prompt-safe MCP server inventory, refreshed/
  failed/partial servers, transports, tool/resource/prompt counts, and the last
  inventory refresh records.
- `SkillCenterTrace` summarizes loaded skills and windowed skill resource views
  without embedding skill bodies into the trace summary.
- `HandoffTrace` summarizes multi-agent handoff decisions, selected sessions,
  source sessions, candidate counts, and denial/not-found status.
- `AgentToolTrace` summarizes child-agent calls made through the normal tool
  loop, including target sessions, child run ids, completion/failure counts,
  and output/task byte counts.
- `TraceCorrelationIndex` cross-references provider calls, tool replay records,
  policy decisions, approvals, event log entries, and journal replay events by
  run, turn, call id, approval id, decision id, and subject.
- `RunTraceStorePort` plus `InMemoryRunTraceStore`, `SQLiteRunTraceStore`, and
  `MarkdownRunTraceStore` for lightweight durable trace capture.
- `RunTraceQuery` gives those stores one portable filter surface for status,
  run ids, metadata equality, limits, and reverse ordering.
- `AgentSession.trace_store` lets `AgentRunner` persist trace bundles
  automatically after a run completes.
- Runtime code can persist the bundle in PG, object storage, logs, or a workflow
  database without changing SDK execution semantics.

### Replay And Eval

- `TraceReplayHarness` builds a deterministic replay timeline from journal,
  event-log, and provider call manifests, including resume-plan,
  checkpoint-loaded, preflight, prompt-budget, prompt-bucket-budget,
  prompt-semantic-trim, prompt-trim, context-material-selection, MCP inventory/server,
  skill-load/resource-view, handoff decision, approval request/decision,
  artifact-store, structured-output validation/repair, provider-call,
  provider request-shape payloads, provider-stream, embedding-call, and
  lifecycle-hook steps. Provider-native tool results are replayed as explicit
  derived steps when journal tool-call records carry `provider_tool_call`
  metadata.
- `TraceReplayComparator` compares two trace manifests and reports deterministic
  replay diffs for regression baselines.
- `TraceEvalSpec` defines provider-neutral expectations such as status,
  iteration limits, provider call limits, required provider names/models,
  required provider model capabilities, provider stream event contracts,
  provider-native tool-call presence and tool-call names,
  provider/tool error classification kinds,
  unified failure-summary sources/kinds,
  provider route plans and provider route preflight readiness,
  embedding call limits, required embedding providers/models/dimensions,
  required events, required tools, preflight presence/status/issue-code constraints,
  resume-plan presence, resume-plan readiness,
  expected checkpoint id, handoff status/session constraints, tool execution presence, tool retry, tool schema
  validation, minimum tool attempt counts, ToolCenter route/call audit
  constraints, agent-as-tool session/status constraints, MCP center inventory constraints, skill center constraints,
  approval status/subject constraints, artifact count/size/type constraints,
  structured output schema/repair/failure constraints, storage backend constraints,
  storage backend preflight readiness constraints,
  lifecycle hook constraints, event-log presence, event-log types, terminal
  events, event sequence monotonicity, duplicate sequence limits,
  context injection name/source/target/status constraints, included/trimmed/
  excluded injection source constraints, context material selection constraints,
  memory governance constraints, prompt
  budget constraints, bucket budget constraints, runtime semantic prompt trim
  constraints, provider request-shape constraints, provider-native tool result
  constraints, and global prompt trim constraints.
- `DefaultTraceEvaluator` evaluates one trace manifest without calling a model.
- `TraceEvalHarness` evaluates traces from any `RunTraceStorePort`.
- `TraceEvalReport` exports replay, summary counters, and contract failures.
- `TraceEvalCase`, `TraceEvalSuite`, and `TraceEvalSuiteRunner` group multiple
  run IDs and per-case specs into one provider-neutral regression report.

The SDK owns trace replay shape, deterministic trace diffs, and generic
run-level evaluation mechanics. The SDK also owns the generic multi-case suite
runner. Runtimes own domain-specific datasets, eval suites, baseline selection,
product dashboards, scoring policy, and regression data retention.

### Human-In-Loop Approvals

- `PolicyDecisionStorePort` records each action/tool policy gate decision for
  replay, eval, and audit.
- `InMemoryPolicyDecisionStore`, `SQLitePolicyDecisionStore`, and
  `MarkdownPolicyDecisionStore` provide lightweight SDK-local policy audit
  backends.
- `ApprovalRequest` manifests produced by policy decisions.
- `ApprovalStorePort` for approval queues.
- `InMemoryApprovalStore` for tests and lightweight runtimes.
- `SQLiteApprovalStore` and `MarkdownApprovalStore` for durable local or
  inspectable SDK runs.
- `NullApprovalStore` for runtimes that only need per-run approval metadata.
- `ApprovalRecord` and `ApprovalDecisionRecord` manifests for audit/replay.
- `ApprovalTrace` and trace eval contracts for required approval statuses,
  subjects, subject kinds, approved subjects, pending limits, and rejected
  limits.
- `ApprovalCenter`, `ApprovalQueueView`, and `ApprovalResolution` for queue
  inspection, approve/reject/cancel helpers, and resume-context generation.
- `ApprovalResumeContext` for passing approved decisions into resumed runs.
- `approval_resumed` events when a matching approved subject unlocks an
  action/tool policy gate.

The SDK owns policy decision records, approval state contracts, queue/resolution
manifests, resume context generation, and event emission. Runtimes own the
organization policy source, operator UI, identity, authorization, notification
routing, SLA policy, SIEM export, and durable workflow storage.

### Context Reduction

- `ContextReducerPort` for automatic timeline/context compaction.
- `DefaultContextReducer` for deterministic local reduction.
- `ReducerRequest` and `ReducerResult` manifests for replay and audit.
- Pinned item retention plus recent-window retention.
- `apply_reduction_to_timeline()` for updating a `TimelineStore` with compressed
  head text and archive refs.
- Optional `AgentSession.context_reducer` integration. When configured,
  `AgentRunner` reduces over-budget timeline state before building `PromptIR`,
  then exposes the reduction manifest on both `AgentRunOutcome` and prompt
  metadata.

The SDK owns reduction mechanics and manifests. Runtimes may replace the reducer
with an LLM summarizer, vector/archive backed compressor, or domain-specific
reducer while keeping the same core contract.

### Embeddings And Semantic Ranking

- `EmbeddingProviderPort` protocol.
- `EmbeddingProviderCenter` for provider-neutral embedding routing and call audit.
- `EmbeddingRequest`, `EmbeddingResponse`, `EmbeddingVector`, and
  `EmbeddingRoute` manifests.
- Embedding call manifests can be attached to run traces and evaluated/replayed
  with the same provider-neutral harness as LLM calls.
- `DeterministicEmbeddingProvider` for dependency-free tests and lightweight
  local semantic ranking.
- `rank_semantic_documents()` for embedding-backed ranking over SDK search documents.
- `InMemoryMemoryStore` can optionally use an embedding provider for
  `semantic` and `hybrid` memory search.

Concrete embedding clients for OpenAI, Gemini, local models, vector services, or
product gateways live in runtimes or adapter packages. The SDK owns the portable
request/response/route shape, call manifests, deterministic fallback, and the
ranking contract used by lightweight core stores.

### LLM Providers

- `LLMProviderPort` protocol.
- `LLMTransportPort`, `LLMProviderCodecPort`, `DefaultLLMProviderCodec`, and
  `TransportLLMProvider` for dependency-free provider adapter contracts.
- `OpenAICompatibleLLMProviderCodec` for Chat Completions-style payload
  encoding/decoding without importing the OpenAI SDK.
- `AgentCoreProviderAcceptanceHarness` for no-network provider contract checks,
  including default transport payloads, prompt-safe request manifests,
  OpenAI-compatible streamed tool calls, and retryable stream errors.
- Provider registry and routing.
- `LLMContentPart` for provider-neutral text, image, audio, file, binary, and
  JSON message parts without binding the SDK to one vendor message schema.
- `LLMToolContract`, `LLMToolChoice`, and `LLMResponseFormat` for
  provider-neutral native tool/function calling and JSON/JSON-schema response
  contracts.
- `LLMToolContract.from_tool_spec()` for deriving native model tool contracts
  from the same `ToolSpec` schema used by the SDK tool center.
- `LLMToolCall` for provider-neutral model-requested tool calls, including
  prompt-safe manifests and transport payloads.
- `LLMModelCapabilities` for provider-neutral model context windows, output
  limits, streaming support, tool-call support, JSON mode, structured output,
  and modality declarations.
- `LLMRequestShapePlan` for provider-aware request shaping before calls, such
  as capping `max_output_tokens` to a declared model output limit.
- `LLMRetryPolicy` for retry/fallback behavior.
- `LLMUsageLimits` for cost, call-attempt, and token budgets.
- Usage/failure accounting.
- Budget checks.
- Request, response, stream event, route, and call record manifests.
- `LLMProviderRoutePlan` for preflight route explanations before any provider
  call, including selected provider, unsupported candidates, and fallback
  candidates.
- `LLMStreamAccumulator` for reconstructing a stream into an `LLMResponse`
  plus prompt-safe stream summaries.
- Provider call audit records for completed and failed attempts, including the
  routed request manifest, original request manifest, and route plan manifest.
- Failed provider calls include prompt-safe `ErrorClassification` manifests for
  common retry/failure reporting.
- Stream error events are treated as failed attempts for retry/fallback audit.
- Streaming attempts are budget-checked before events are emitted by the center,
  and completed streamed calls record a standard stream summary.
- Stream summaries expose event types and error counts to trace/eval contracts
  without storing raw streamed content.
- `LLMUsageLimits` and `LLMProviderCenter` enforce SDK-local call-attempt,
  cost, and token budgets before and after provider execution.
- `AgentSession.stream` and `AgentRunRequest.stream` let runners choose
  streaming per session or per run without changing provider adapters.

Concrete clients for OpenAI, Anthropic, local models, gateways, credentials, and
vendor rate limits live in runtimes or adapter packages. `agent_core` owns the
provider-neutral request/response/stream shapes, stream aggregation, and
error/retry mapping. When runtimes declare `LLMModelCapabilities`,
`LLMProviderCenter` can avoid routes that cannot satisfy requested model
features and can shape compatible requests to declared output limits before an
adapter sees them.
Native tool calling and response-format contracts are represented as SDK
request fields, not vendor payloads. Adapter packages translate them to
OpenAI-compatible `tools`/`response_format`, Anthropic tool use, Gemini function
calling, local model schemas, or gateway-specific payloads.
For OpenAI-compatible HTTP gateways, `OpenAICompatibleLLMProviderCodec` provides
the common message/tool/response-format/usage/tool-call mapping while the
runtime still owns HTTP, auth, retries at the network layer, and deployment
selection.
`ReActConfig.native_tool_calls` lets a runtime opt into the provider-native
loop: the executor sends `ToolSpec`-derived contracts, executes returned
`LLMToolCall` items through the same policy/replay/tool boundary, and appends
provider-neutral tool result messages for the next model turn.
Journal records keep the originating `provider_tool_call` together with the
prompt-safe tool result metadata, so replay/eval can prove provider-native tool
calls were actually executed and returned to the model loop.
Content parts follow the same rule: the SDK can describe text, image, audio,
file, binary, and JSON parts, while runtimes own file access, uploads, URL
signing, object storage, and vendor-specific multipart payloads.
The routed request manifest records the selected provider, model priority, and
model capabilities so trace/eval contracts can later prove which model ability
was actually used for structured output, JSON mode, tool calls, streaming, or
modalities.
The request shape plan records whether `max_output_tokens` was capped, the
original and final output limit, provider/model identity, and declared context
window. This keeps provider limits auditable without importing vendor SDKs.
Trace replay includes the request shape plan in provider-call payloads, and
trace evals can require adjusted plans, specific decisions, provider names, or
final output-token ceilings.
The route plan manifest records the full provider preflight decision, so a
runtime can explain why a cheap/default model was skipped, which provider was
selected, and which compatible providers remain available for fallback without
calling any concrete model client.

### Tools

- `ToolSpec`, `ToolRegistry`, and `ToolCenter`.
- Runtime mounts.
- Tool tags, aliases, manifests, inventory, and search.
- `ToolRoutePlan`, `ToolRouteCandidate`, and `ToolCenterCallRecord` for
  prompt-safe preflight routing and call audit across mounted tool runtimes.
- `ToolCenterTrace` for run-level route/call summaries and eval contracts.
- `ToolExecutionCenter`, `ToolRetryPolicy`, and tool execution attempt records
  for retryable failure recovery and audit.
- Tool execution attempts and final failed results include prompt-safe
  `ErrorClassification` manifests for schema, exception, and tool-result
  failures.
- `validate_tool_arguments()` and pre-execution schema validation for
  `ToolSpec.parameters_schema`.
- Tool schema failures produce prompt-safe `SchemaValidationResult` manifests
  and do not call the underlying runtime.
- In-memory replay.
- `ToolReplayRecord` manifests for deterministic replay audit.
- `ToolReplayStorePort` and `PersistentToolReplay` for pluggable replay storage.
- In-memory, SQLite, and Markdown replay stores for lightweight SDK use.
- Invocation manifests hash arguments instead of exposing full argument values.

### Artifacts

- `ArtifactStorePort` for large prompt-unsafe observations.
- `InMemoryArtifactStore`, `SQLiteArtifactStore`, and `MarkdownArtifactStore`
  for lightweight SDK use.
- `ReActExecutor` can store oversized tool results as artifacts and place only a
  prompt-safe artifact reference in the next model message.
- `ArtifactTrace` and trace eval contracts for artifact kinds, tool ownership,
  content types, count limits, total-byte limits, and per-artifact size limits.

### Skills

- `SkillRegistry` and `SkillsContext`.
- Markdown skill parsing.
- `SKILL.md` discovery.
- Zip skill archive loading with path-safety checks.
- Resource loading and windowed context views.
- `SkillCenterTrace` and trace eval contracts for loaded skills and resource
  view windows.

### MCP

- `MCPCenter`.
- Tool, resource, and prompt registration.
- Server state refresh.
- `refresh_inventory()` refreshes MCP tools, resources, and prompts in one
  auditable pass and returns per-server prompt-safe inventory manifests.
- `MCPContextMaterialRequest` and `MCPCenter.context_materials()` turn MCP
  resources/prompts into selector-ready `ContextMaterial` without making MCP a
  product-specific runtime adapter.
- `AgentRunRequest.mcp_context_materials` lets `AgentRunner` explicitly collect
  MCP resources/prompts and send them through the normal context material
  selector before prompt assembly.
- `MCPCenterTrace` and trace eval contracts for required servers, refreshed
  servers, forbidden statuses, failed-server limits, and partial-refresh limits.
- SDK-free stdio JSON-RPC connector.
- `CapabilityQuery`, `CapabilityMatch`, and `CapabilityDiscoveryResult` for
  unified discovery across actions, local tools, skills, MCP tools, MCP
  resources, MCP prompts, and MCP servers.
- Discovery manifests are prompt-safe and traceable; runtimes own concrete MCP
  sessions, credentials, UI, and network/process lifecycle.

### Memory

- `MemoryCenter` and `MemoryPort`.
- `ExternalMemoryStore` and `MemoryCenter.register_spec()` for runtime-owned
  PG/vector/graph/product memory adapters without adding database drivers to core.
- `ExternalMemoryCallRecord` for prompt-safe audit of runtime-owned memory
  adapter search/write calls, including backend kind, query/write manifests,
  hit counts, status, and errors.
- `MemoryQuery.mode` for keyword, semantic, vector, graph, and hybrid recall.
- `MemoryStoreSpec.backend_kind`, namespace, and capability flags for
  Postgres/vector/graph/product stores implemented outside core.
- `MemorySearchPlan` and `MemoryRoute` for prompt-safe, traceable store routing.
- In-memory memory store.
- SQLite memory store.
- Markdown memory store.
- Postgres, vector DB, graph, or product memory can implement `MemoryPort`
  outside core and publish backend metadata plus call audit through
  `ExternalMemoryStore`.
- Runner-level memory recall injection through `ContextInjection`.
- Memory governance hooks.
- Leak scanning and global bucket checks.
- Prompt-safe governance manifests for allow/rewrite/deny write decisions.

### Data Backend Boundary

The SDK core treats data backends as ports, not as product commitments:

| Data area | Core port | Built-in lightweight implementations | External/runtime implementations |
| --- | --- | --- | --- |
| Memory | `MemoryPort` | In-memory, SQLite, Markdown | Postgres, vector DB, graph/RAG, product knowledge stores |
| Context material | `ContextMaterialStorePort` | In-memory, SQLite, Markdown | Postgres, vector DB, graph/RAG, product APIs |
| Harness journal | `AgentJournalStorePort` | In-memory snapshot store, SQLite snapshot store, Markdown snapshot store | Postgres, object storage, event log, workflow database |
| Tool replay | `ToolReplayStorePort` | In-memory, SQLite, Markdown | Postgres, object storage, workflow replay DB |
| Run traces | `RunTraceStorePort` | In-memory, SQLite, Markdown | Postgres, object storage, observability pipeline |
| Manager runs | `AgentRunStorePort` | In-memory, SQLite, Markdown | Postgres, workflow DB, scheduler state |
| Planner state | `PlannerStorePort` | In-memory, SQLite, Markdown | Postgres, workflow DB, planner audit store |
| Artifacts | `ArtifactStorePort` | In-memory, SQLite, Markdown | Filesystem, object storage, build artifacts |
| Approvals | `ApprovalStorePort` | Null, in-memory, SQLite, Markdown | Approval service, ticketing/workflow DB, operator UI |
| Events | `EventSinkPort`, `EventLogPort` | In-memory, SQLite, Markdown | UI stream, logs, metrics, audit pipeline |

This keeps `agent_core` small and importable while still leaving a clean path to
production storage. A runtime should bring its own durable backends when it needs
PG, graph memory, vector indexing, multi-tenant isolation, retention policy, or
product audit. The SDK still owns the portable route/query manifest so these
backends can be swapped without changing ReAct or harness semantics.

Every core store manifest includes a shared `backend` block using
`agent-core-storage-backend/v1`. It records the data role, backend kind,
durability, inspectability, queryability, transaction support, and whether the
backend is a core builtin. Runtime-owned PG, vector, graph, object-store, or
product backends should export the same shape while living outside this
repository.

`AgentStateBundleBuilder` / `build_agent_state_bundle` turns generic SDK state
manifests into a prompt-safe portable bundle for migration, archive, and
recovery preflight. Bundle components record role, backend kind, schema version,
byte counts, SHA-256 digests, redaction decisions, and a role-ordered restore
plan. Concrete copying between SQLite, Markdown, PG, object storage, workflow DB,
or product stores remains a runtime/adapter job.

`StorageBackendCatalog` registers those manifests and preflights backend
selection through `StorageBackendRequirement`. It can select by role, allowed
kind, namespace, durability, queryability, transaction support, inspectability,
and required capabilities, then return a prompt-safe `StorageBackendSelection`
with every candidate and rejection reason. This is the SDK-level contract for
SQLite/Markdown/PG/vector/graph choices; concrete drivers stay in runtime or
adapter packages.
For multi-role startup checks, `StorageBackendCatalog.preflight()` returns a
`StorageBackendPreflightReport` across requirements such as memory, context
material, run traces, events, and artifacts. Trace replay/eval can require that
report, assert it is ready, verify covered roles, cap blocking selections, and
forbid rejection reasons such as missing capabilities.
`storage_backend_manifests_from_components()` and
`storage_backend_catalog_from_components()` can build the catalog directly from
component manifests such as `MemoryCenter`, `ContextMaterialCenter`, and trace
bundles. This lets runtimes publish PG/vector/graph stores through normal SDK
manifests and still preflight them without importing adapter code.

### Planner

- `PlannerPort` for plan-and-execute agents.
- `Plan`, `PlanStep`, and `PlanUpdate` state contracts.
- `InMemoryPlanner` for tests, examples, and lightweight embeddings.
- `PersistentPlanner` backed by `PlannerStorePort`.
- `InMemoryPlannerStore`, `SQLitePlannerStore`, and `MarkdownPlannerStore`.
- Dependency-aware ready-step selection.
- `PlanExecutor` for sequential ready-step execution through `AgentSessionManager`.
- `PlanExecutionReport` and `PlanExecutionStep` manifests for audit.
- `PlannerTrace` plus replay/eval contracts for plan ids, step status, and execution reports.
- Manifest export with terminal state, ready steps, and status counts.

The SDK owns generic plan state, update semantics, sequential step execution,
single-process manager capacity guards, local pending queues, and
schedule/capacity snapshots.
Runtimes own the actual planning strategy: Raven can produce pentest plans, a
code agent can produce repair plans, and an ops agent can produce incident
response plans while all of them reuse the same core contract. Runtime code also
owns distributed scheduling, cross-worker concurrency, retry policy, and
business workflow rules.

### Prompt Buckets

`agent_core` owns prompt structure, not product-specific prompt wording. Runtime
packages provide business instructions; the SDK provides bucket ordering,
context injection, trimming, hashing, and replay manifests.

| Bucket | Purpose |
| --- | --- |
| Static | Stable role, rules, and agent contract |
| Capability | Available tools, MCP tools, actions, and skills |
| Skill | Loaded skill bodies and resource windows |
| Memory | Recalled facts, continuity hints, prior observations |
| Task | Current objective, constraints, and runtime context |
| Reactive | Recent tool results, failures, deltas, loop-local state |

`PromptIR.trim_plan()` and `PromptIR.trim_to_budget()` use
`DEFAULT_PROMPT_TRIM_RULES` to produce an auditable `PromptTrimPlan` before
cutting prompt text. The default rules trim volatile timeline context first,
then semi-dynamic recall/schema material, then capability inventory, while
protecting high-static system rules and preserving a minimum current-task
window. The final prompt manifest records `PromptTrimResult` steps, removed
bytes, protected roles, and convergence. `AgentRunner` applies this against
the effective prompt budget before calling the provider. `AgentPromptBudgetPlan`
derives that budget from `RuntimeBudget.max_prompt_bytes` and, when available,
provider route capabilities such as context window and reserved output tokens.

If `AgentSession.context_reducer` is configured, `AgentRunner` first applies
timeline reduction against `RuntimeBudget.max_timeline_bytes`. This gives the SDK
Yaklang-style automatic context compaction without forcing a Raven-specific
summarizer or storage backend into core.

`PromptBucketBudgetPolicy` can apply per-bucket caps before global prompt
trimming. For example, a code agent can cap recalled memory while preserving the
current task, and a security runtime can cap capability inventory while keeping
high-static rules protected. The result is recorded in prompt and run trace
manifests as `agent-core-prompt-bucket-budget-result/v1`.

`PromptSemanticReducerPort` is the runtime-time semantic trimming hook. It runs
after bucket-local budgets and before the final byte-budget trim. The SDK
provides `PromptSemanticTrimRequest`, `PromptSemanticTrimResult`, and
`DefaultPromptSemanticReducer` for deterministic local behavior; production
runtimes can swap in embedding-backed, LLM-backed, or domain-specific reducers
without changing `AgentRunner`. Prompt and trace manifests record the reducer
request hash, per-bucket decisions, selected/dropped unit counts, convergence,
and trimmed count without storing hidden raw context outside the prompt itself.

`ContextInjection` lets runtimes or core services place structured material into
a target bucket without rewriting the prompt builder. Resume checkpoints use this
path today. `AgentRunner` also injects memory recall through this path when
memory is enabled; operator hints and runtime-specific context can use the same
SDK-level mechanism later.

`ContextMaterial`, `ContextMaterialSelectionRequest`, and
`DefaultContextMaterialSelector` provide the earlier selection step for
runtime-provided candidate context. A runtime can collect candidates from
SQLite, Markdown, PG, vector search, graph memory, tools, or product APIs, then
let the SDK rank them by deterministic task-term overlap plus priority and
budget. The selector returns `ContextInjection` objects and a prompt-safe
manifest with selected/dropped status, target buckets, scores, byte counts, and
hashes.
`AgentSession.context_material_selector` lets `AgentRunner` apply that selector
automatically when `AgentRunRequest.context_materials` or
`AgentRunRequest.context_material_selection` is provided.
`ContextMaterialCenter` and `ContextMaterialStorePort` add a backend-neutral
candidate context layer. The SDK ships in-memory, SQLite, and Markdown stores
plus an external-store wrapper with prompt-safe call audit; runtimes can mount
PG, vector, graph, or product stores without changing runner/ReAct code.
`AgentRunRequest.context_material_query` makes runner collection explicit and
feeds store results into the same selector before prompt assembly.
`AgentRunRequest.mcp_context_materials` uses the same path for MCP
resources/prompts: runner collection is opt-in, selector/budget policy remains
SDK-owned, and concrete MCP process/auth/network lifecycle remains runtime-owned.

`ContextInjectionPolicy` governs those insertions before prompt assembly. It can
limit allowed target buckets, trim oversized injected material, cap total
injection bytes, and record included/excluded/trimmed decisions in the prompt
manifest. `AgentSession.context_injection_policy` lets runners apply the same
policy to resume, memory, approval, and runtime-provided injections.

`ContextWindowBuilder` / `build_context_window_report` converts prompt or run
trace manifests into one prompt-safe context window audit. It rolls up selected,
dropped, injected, trimmed, and excluded context entries, bucket byte counts,
semantic trim metadata, and bucket-budget metadata. `ContextWindowPolicy` can
flag over-budget prompts, excessive exclusions/trims, or missing required
context sources without depending on Raven, Graphiti, or a concrete runtime
ranking model.

## Yaklang Influence

This project studies Yaklang-style agent ideas at the architecture level:

- Semantic context partitioning.
- Automatic context trimming.
- Context injection.
- Memory-backed continuity.
- Tool, skill, and MCP capability orchestration.

It does not translate Yaklang source code. Yaklang is AGPL-licensed; this repository is an independent implementation of similar ideas.

## Repository Layout

```text
agent_core/        # standalone SDK package
tests/             # core tests only
examples/          # pure core examples, no runtime adapters
docs/              # architecture notes
.github/workflows/ # CI
```

## Install

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
```

On Linux/macOS:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

The core package currently has no required runtime dependencies.

## Test

```bash
python -m pytest
```

The expected baseline is the full repository test suite passing.

## Examples

Run a deterministic ReAct session:

```bash
python examples/minimal_react.py
```

Run memory and skill context examples:

```bash
python examples/memory_and_skills.py
```

## How A Runtime Uses This SDK

A runtime should:

1. Implement or provide an `LLMProviderPort`.
2. Register tools through `ToolRegistry` or mount a `ToolRuntimePort`.
3. Load skills through `SkillRegistry` / `SkillsContext`.
4. Attach memory through `MemoryPort` / `MemoryCenter`.
5. Build an `AgentSession`.
6. Run it with `AgentRunner`.
7. Resume from checkpoints by passing a `ResumeToken` into `AgentRunRequest`.
8. Optionally set `AgentRunRequest.stream` for streaming provider runs.
9. Optionally set `AgentRunRequest.timeout_seconds` or cancel background runs
   through `AgentSessionManager.cancel()`.

The runtime may be Raven, a code agent, an ops agent, or any other host. The runtime owns concrete tools, credentials, persistence, UI, and deployment. `raven_heart` owns the reusable agent mechanics.

## Current Status

| Area | Status |
| --- | --- |
| Harness/ReAct core | MVP implemented |
| Task contract profile | MVP implemented |
| Provider center | MVP implemented |
| Provider call audit | MVP implemented |
| Provider contract profile | MVP implemented |
| Provider transport contract | MVP implemented |
| OpenAI-compatible provider codec | MVP implemented |
| Provider route plan/preflight audit | MVP implemented |
| Run provider route preflight gate | MVP implemented |
| Provider model capabilities | MVP implemented |
| Provider tool/response-format contracts | MVP implemented |
| Provider-native tool-call loop | opt-in MVP implemented |
| Runner streaming control | MVP implemented |
| Provider multimodal content contracts | MVP implemented |
| Embedding provider center | MVP implemented |
| Semantic ranking contract | MVP implemented |
| Embedding trace/eval contracts | MVP implemented |
| Schema validation contracts | MVP implemented |
| Trace correlation | MVP implemented |
| Trace observability manifests | MVP implemented |
| Run failure summary trace/eval | MVP implemented |
| SDK capability/boundary manifest | MVP implemented |
| SDK replacement-readiness profile | MVP implemented |
| SDK public API stability contract | MVP implemented |
| SDK API lifecycle policy | MVP implemented |
| SDK runtime boundary audit | MVP implemented |
| SDK packaging acceptance harness | MVP implemented |
| SDK replacement acceptance harness | MVP implemented |
| SDK context acceptance harness | MVP implemented |
| SDK provider acceptance harness | MVP implemented |
| SDK provider conformance harness | MVP implemented |
| SDK budget acceptance harness | MVP implemented |
| SDK redaction acceptance harness | MVP implemented |
| SDK eval suite runner | MVP implemented |
| SDK task profile acceptance harness | MVP implemented |
| SDK recovery acceptance harness | MVP implemented |
| SDK durable session acceptance harness | MVP implemented |
| SDK resume acceptance harness | MVP implemented |
| SDK coordination acceptance harness | MVP implemented |
| SDK event acceptance harness | MVP implemented |
| SDK external backend acceptance harness | MVP implemented |
| SDK guardrail acceptance harness | MVP implemented |
| SDK lifecycle acceptance harness | MVP implemented |
| SDK native tool acceptance harness | MVP implemented |
| SDK aggregate validation suite | MVP implemented |
| Run trace query | MVP implemented |
| Manager run state query | MVP implemented |
| Manager run event paging | MVP implemented |
| Manager run event tail/poll | MVP implemented |
| Lifecycle hook contracts | MVP implemented |
| Lifecycle hook trace/eval contracts | MVP implemented |
| Run preflight guardrails | MVP implemented |
| Run preflight trace/eval contracts | MVP implemented |
| Run storage backend preflight gate | MVP implemented |
| Agent-as-tool runtime | MVP implemented |
| Agent-as-tool trace/eval contracts | MVP implemented |
| Handoff trace/eval contracts | MVP implemented |
| Interrupt/cancel/timeout semantics | MVP implemented |
| Tool center | MVP implemented |
| Tool center route/call audit | MVP implemented |
| Tool center trace/eval contracts | MVP implemented |
| Tool replay store | MVP implemented |
| Skill center | MVP implemented |
| MCP center | MVP implemented |
| MCP context material export | MVP implemented |
| Capability discovery | MVP implemented |
| SDK orchestration acceptance harness | MVP implemented |
| SQLite/Markdown memory | MVP implemented |
| Memory backend routing/specs | MVP implemented |
| External memory call audit | MVP implemented |
| Memory governance trace/eval | MVP implemented |
| Unified storage backend manifests | MVP implemented |
| Storage backend catalog/selection | MVP implemented |
| SDK storage acceptance harness | MVP implemented |
| Planner core | MVP implemented |
| Planner trace/eval contracts | MVP implemented |
| Approval core | MVP implemented |
| SDK approval acceptance harness | MVP implemented |
| Approval trace/eval contracts | MVP implemented |
| Artifact trace/eval contracts | MVP implemented |
| Structured output trace/eval contracts | MVP implemented |
| Prompt buckets/trimming | semantic trim plan MVP |
| Provider-aware prompt budget | MVP implemented |
| Provider-aware prompt budget replay/eval | MVP implemented |
| Prompt bucket budget policy | MVP implemented |
| Runtime semantic prompt reducer | MVP implemented |
| Runtime semantic prompt trace/eval | MVP implemented |
| Context material selector | runner-integrated MVP |
| Context material store center | MVP implemented |
| Context material selection trace/eval | MVP implemented |
| Context injection policy | MVP implemented |
| Context reducer | runner-integrated MVP |
| Runtime adapter code | intentionally excluded |
| Full OpenAI Agents SDK replacement | in progress |

## License

MIT. See `LICENSE`.
