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
- Context injection records for resume, memory, runtime hints, and other bucketed material.
- Context injection policy for bucket allow-lists, per-injection trimming, total injection budget, and audit manifests.
- Provider-neutral embedding request/response, provider routing, and semantic ranking contracts.
- SQLite, Markdown, and in-memory stores for lightweight memory.
- Memory governance decisions for allow/rewrite/deny write auditing.
- Unified storage backend manifests for core stores and runtime-owned backends.
- External memory store wrapper for runtime-owned PG/vector/graph/product adapters.
- Pluggable journal stores for harness checkpoint/resume persistence.
- Journal replay manifests and snapshot consistency audit.
- Run trace bundle for provider/tool/approval/journal/event audit aggregation.
- Run trace store port plus in-memory, SQLite, and Markdown trace stores.
- Trace replay and evaluation harness for provider-neutral run and embedding audits.
- Manager run state store port plus in-memory, SQLite, and Markdown stores.
- Sequenced event stream plus in-memory, SQLite, and Markdown event logs.
- Planner protocol plus in-memory, SQLite, and Markdown plan state stores.
- Multi-agent handoff specs, routing decisions, and lightweight coordination.
- Human-in-loop approval request and decision queue primitives.
- Approval resume context for approved action/tool gate continuation.
- Policy gates, budget metadata, loop guards, and capability manifests.
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
| Trace replay/eval harness | Product-specific eval suites and dashboards |
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
- `AgentRunStorePort` for manager-level queued/running/completed state.
- `InMemoryAgentRunStore`, `SQLiteAgentRunStore`, and `MarkdownAgentRunStore`
  for lightweight run state persistence.
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
- `MCPCenterTrace` summarizes prompt-safe MCP server inventory, refreshed/
  failed/partial servers, transports, tool/resource/prompt counts, and the last
  inventory refresh records.
- `SkillCenterTrace` summarizes loaded skills and windowed skill resource views
  without embedding skill bodies into the trace summary.
- `HandoffTrace` summarizes multi-agent handoff decisions, selected sessions,
  source sessions, candidate counts, and denial/not-found status.
- `TraceCorrelationIndex` cross-references provider calls, tool replay records,
  policy decisions, approvals, event log entries, and journal replay events by
  run, turn, call id, approval id, decision id, and subject.
- `RunTraceStorePort` plus `InMemoryRunTraceStore`, `SQLiteRunTraceStore`, and
  `MarkdownRunTraceStore` for lightweight durable trace capture.
- `AgentSession.trace_store` lets `AgentRunner` persist trace bundles
  automatically after a run completes.
- Runtime code can persist the bundle in PG, object storage, logs, or a workflow
  database without changing SDK execution semantics.

### Replay And Eval

- `TraceReplayHarness` builds a deterministic replay timeline from journal,
  event-log, and provider call manifests, including resume-plan,
  checkpoint-loaded, prompt-bucket-budget, prompt-semantic-trim,
  prompt-trim, context-material-selection, MCP inventory/server,
  skill-load/resource-view, handoff decision, approval request/decision,
  artifact-store, structured-output validation/repair, provider-call,
  provider-stream, embedding-call, and lifecycle-hook steps.
- `TraceReplayComparator` compares two trace manifests and reports deterministic
  replay diffs for regression baselines.
- `TraceEvalSpec` defines provider-neutral expectations such as status,
  iteration limits, provider call limits, required provider names/models,
  required provider model capabilities, provider stream event contracts,
  provider-native tool-call presence and tool-call names,
  embedding call limits, required embedding providers/models/dimensions,
  required events, required tools, resume-plan presence, resume-plan readiness,
  expected checkpoint id, handoff status/session constraints, tool execution presence, tool retry, tool schema
  validation, minimum tool attempt counts, ToolCenter route/call audit
  constraints, MCP center inventory constraints, skill center constraints,
  approval status/subject constraints, artifact count/size/type constraints,
  structured output schema/repair/failure constraints, storage backend constraints,
  lifecycle hook constraints, event-log presence, event-log types, terminal
  events, event sequence monotonicity, duplicate sequence limits,
  context injection name/source/target/status constraints, included/trimmed/
  excluded injection source constraints, context material selection constraints,
  memory governance constraints, prompt
  bucket budget constraints, runtime semantic prompt trim constraints, and
  global prompt trim constraints.
- `DefaultTraceEvaluator` evaluates one trace manifest without calling a model.
- `TraceEvalHarness` evaluates traces from any `RunTraceStorePort`.
- `TraceEvalReport` exports replay, summary counters, and contract failures.

The SDK owns trace replay shape, deterministic trace diffs, and generic
run-level evaluation mechanics. Runtimes own domain-specific eval suites,
baseline selection, product dashboards, scoring policy, and regression data
retention.

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
- Stream error events are treated as failed attempts for retry/fallback audit.
- Streaming attempts are budget-checked before events are emitted by the center,
  and completed streamed calls record a standard stream summary.
- Stream summaries expose event types and error counts to trace/eval contracts
  without storing raw streamed content.

Concrete clients for OpenAI, Anthropic, local models, gateways, credentials, and
vendor rate limits live in runtimes or adapter packages. `agent_core` owns the
provider-neutral request/response/stream shapes, stream aggregation, and
error/retry mapping. When runtimes declare `LLMModelCapabilities`,
`LLMProviderCenter` can avoid routes that cannot satisfy requested model
features or declared token/output limits.
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
Content parts follow the same rule: the SDK can describe text, image, audio,
file, binary, and JSON parts, while runtimes own file access, uploads, URL
signing, object storage, and vendor-specific multipart payloads.
The routed request manifest records the selected provider, model priority, and
model capabilities so trace/eval contracts can later prove which model ability
was actually used for structured output, JSON mode, tool calls, streaming, or
modalities.
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

`StorageBackendCatalog` registers those manifests and preflights backend
selection through `StorageBackendRequirement`. It can select by role, allowed
kind, namespace, durability, queryability, transaction support, inspectability,
and required capabilities, then return a prompt-safe `StorageBackendSelection`
with every candidate and rejection reason. This is the SDK-level contract for
SQLite/Markdown/PG/vector/graph choices; concrete drivers stay in runtime or
adapter packages.

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
`RuntimeBudget.max_prompt_bytes` before calling the provider.

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

`ContextInjectionPolicy` governs those insertions before prompt assembly. It can
limit allowed target buckets, trim oversized injected material, cap total
injection bytes, and record included/excluded/trimmed decisions in the prompt
manifest. `AgentSession.context_injection_policy` lets runners apply the same
policy to resume, memory, approval, and runtime-provided injections.

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
8. Optionally set `AgentRunRequest.timeout_seconds` or cancel background runs
   through `AgentSessionManager.cancel()`.

The runtime may be Raven, a code agent, an ops agent, or any other host. The runtime owns concrete tools, credentials, persistence, UI, and deployment. `raven_heart` owns the reusable agent mechanics.

## Current Status

| Area | Status |
| --- | --- |
| Harness/ReAct core | MVP implemented |
| Provider center | MVP implemented |
| Provider call audit | MVP implemented |
| Provider transport contract | MVP implemented |
| OpenAI-compatible provider codec | MVP implemented |
| Provider route plan/preflight audit | MVP implemented |
| Provider model capabilities | MVP implemented |
| Provider tool/response-format contracts | MVP implemented |
| Provider-native tool-call loop | opt-in MVP implemented |
| Provider multimodal content contracts | MVP implemented |
| Embedding provider center | MVP implemented |
| Semantic ranking contract | MVP implemented |
| Embedding trace/eval contracts | MVP implemented |
| Schema validation contracts | MVP implemented |
| Trace correlation | MVP implemented |
| Trace observability manifests | MVP implemented |
| Lifecycle hook contracts | MVP implemented |
| Lifecycle hook trace/eval contracts | MVP implemented |
| Handoff trace/eval contracts | MVP implemented |
| Interrupt/cancel/timeout semantics | MVP implemented |
| Tool center | MVP implemented |
| Tool center route/call audit | MVP implemented |
| Tool center trace/eval contracts | MVP implemented |
| Tool replay store | MVP implemented |
| Skill center | MVP implemented |
| MCP center | MVP implemented |
| Capability discovery | MVP implemented |
| SQLite/Markdown memory | MVP implemented |
| Memory backend routing/specs | MVP implemented |
| External memory call audit | MVP implemented |
| Memory governance trace/eval | MVP implemented |
| Unified storage backend manifests | MVP implemented |
| Storage backend catalog/selection | MVP implemented |
| Planner core | MVP implemented |
| Planner trace/eval contracts | MVP implemented |
| Approval core | MVP implemented |
| Approval trace/eval contracts | MVP implemented |
| Artifact trace/eval contracts | MVP implemented |
| Structured output trace/eval contracts | MVP implemented |
| Prompt buckets/trimming | semantic trim plan MVP |
| Prompt bucket budget policy | MVP implemented |
| Runtime semantic prompt reducer | MVP implemented |
| Runtime semantic prompt trace/eval | MVP implemented |
| Context material selector | runner-integrated MVP |
| Context material selection trace/eval | MVP implemented |
| Context injection policy | MVP implemented |
| Context reducer | runner-integrated MVP |
| Runtime adapter code | intentionally excluded |
| Full OpenAI Agents SDK replacement | in progress |

## License

MIT. See `LICENSE`.
