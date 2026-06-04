# Architecture

`agent_core` is a pure SDK base. It defines ports, manifests, execution loops,
context buckets, memory stores, journal stores, and harness state contracts. It
does not contain runtime-specific adapters.

## Dependency Direction

```text
Raven / code agent / ops agent / other runtime
  imports agent_core
  implements provider/tools/memory/persistence
  builds AgentSession
  runs AgentRunner

agent_core
  imports no runtime packages
```

## Boundary Table

| Core area | Runtime responsibility |
| --- | --- |
| LLM provider center, model capabilities, and call audit | Concrete model clients, credentials, rate limits |
| Tool center | Real tools, shell/file/network access, sandboxing |
| Tool replay records and store port | Durable replay backend, retention, cross-run replay policy |
| MCP center | MCP server deployment, credentials, process lifecycle |
| Memory center and `MemoryPort` | Graphiti, RAG, PG/vector/graph/product memory adapters |
| Memory governance contracts | Product retention policy, tenant rules, operator workflows |
| Harness journal, replay, and `AgentJournalStorePort` | PG/event-log/workflow persistence adapters |
| Run trace bundle | Durable trace export, observability pipeline, retention |
| Trace replay/eval harness | Domain eval suites, dashboards, regression policy |
| Manager run state and `AgentRunStorePort` | Product workflow DB, scheduling, distributed workers |
| Multi-agent handoff protocol | Product queues, distributed workers, business orchestration |
| Prompt buckets, trimming, and injection | Domain-specific prompt material and task contracts |
| Prompt bucket budget policy | Domain-specific budget numbers and release gates |
| Structured output specs and validation lifecycle | Domain schemas, typed business objects, persistence |
| Context reducer protocol and reduction manifests | Domain summarizers, archive stores, retrieval policy |
| Planner protocol and plan state | Domain-specific plan generation and workflow policy |
| Harness/ReAct runner | UI events, API routes, production persistence backend |
| Sequenced event stream | UI rendering, logs, metrics, audit pipeline |
| Policy ports and decision records | Operator approval UX, organization policy, SIEM export |
| Approval store and approval manifests | Identity, approval UI, workflow routing, escalation policy |

## Data Backend Boundary

Core ships lightweight implementations so the SDK can run by itself:

| Data area | Port | Built-in implementations | Runtime implementations |
| --- | --- | --- | --- |
| Memory | `MemoryPort` | In-memory, SQLite, Markdown | PG, vector DB, graph/RAG, product knowledge |
| Harness journal | `AgentJournalStorePort` | In-memory snapshot store, SQLite snapshot store, Markdown snapshot store | PG, object storage, workflow DB, audit event log |
| Tool replay | `ToolReplayStorePort` | In-memory, SQLite, Markdown | PG, object storage, workflow replay DB |
| Policy decisions | `PolicyDecisionStorePort` | Null, in-memory, SQLite, Markdown | PG, SIEM/audit log, workflow DB |
| Run traces | `RunTraceStorePort` | In-memory, SQLite, Markdown | PG, object storage, observability pipeline |
| Manager runs | `AgentRunStorePort` | In-memory, SQLite, Markdown | PG, workflow DB, scheduler state |
| Planner state | `PlannerStorePort` | In-memory, SQLite, Markdown | PG, workflow DB, planner audit store |
| Artifacts | `ArtifactStorePort` | In-memory, SQLite, Markdown | Filesystem, object storage, CI/build artifacts |
| Approvals | `ApprovalStorePort` | Null, in-memory, SQLite, Markdown | Approval service, ticketing/workflow DB, operator UI |
| Events | `EventSinkPort`, `EventLogPort` | In-memory, SQLite, Markdown | UI stream, logs, metrics, audit pipeline |

Structured output is intentionally contract-based rather than provider-specific.
`StructuredOutputSpec` can be attached to an `AgentRunRequest`; the runner injects
the schema into prompt context, and `ReActExecutor` validates `finish.output`.
Invalid output becomes repair feedback inside the ReAct loop. Runtime code owns
the domain schema, typed object mapping, and downstream storage.
`StructuredOutputTrace` is derived from journal checkpoints so replay/eval suites
can prove which schemas passed, when repair was requested, and when validation
failed without persisting final-output bodies in trace summaries.
The shared `validate_json_schema_subset()` contract returns prompt-safe
`SchemaValidationResult` manifests and is also used by action argument
validation, so runtime UIs and trace/eval tooling can inspect schema failures
without catching SDK exceptions or depending on a vendor tool-call format.

The SDK must not require a production database driver. Production backends should
be added by the host runtime or by separate adapter packages that implement the
same ports.

`StorageBackendSpec` and `storage_backend_manifest()` give every SDK store and
runtime-owned store one shared backend manifest shape. The manifest records the
store role, backend kind, durability, inspectability, queryability,
transactionality, location, capabilities, and whether the implementation is a
core builtin. Built-in stores use `in_memory`, `sqlite`, `markdown`, or `none`;
runtime adapters can use `postgres`, `vector`, `graph`, `object_storage`,
`product`, `external`, or `custom` without changing ReAct, harness, replay, or
manager code.

`StorageBackendCatalog` is the SDK boundary for choosing among those backends.
It registers built-in and runtime-owned backend manifests, evaluates
`StorageBackendRequirement` constraints, and returns a
`StorageBackendSelection` with candidate scores and rejection reasons. A runtime
can therefore preflight "memory must be tenant-a, durable, semantic, vector
capable" without putting a PostgreSQL, graph, or vector client inside
`agent_core`.

`AgentJournalReplay` turns a journal snapshot into a replayable event manifest
and reports consistency issues before a runtime depends on that state for UI,
debugging, audit, or resume decisions.

`ResumeCandidate` and `ResumeIndex` turn the latest checkpoint per run into a
standard manifest with a `ResumeToken`, checkpoint state, terminal status, and
run metadata. `ResumePlan` adds the preflight decision layer: selected
checkpoint, ready/unavailable status, terminal-run warnings, missing-candidate
errors, terminal-run rejection when `allow_terminal` is false, and a small
summary manifest that can be persisted with manager run state.
`AgentRunner.resume()` and `AgentSessionManager.resume()` convert those
plans into `AgentRunRequest` objects without runtime-side token assembly. The
SDK owns discovery, serialization, local resume entrypoints, and preflight
status; runtime code owns worker selection, user-facing recovery flows, and
distributed resume scheduling.

`AgentRunTraceBundle` aggregates per-run session, prompt, journal replay,
provider audit, tool replay, approvals, event log, resume, timeline reduction,
capability discovery, memory recall/search, ToolCenter, MCP center, skill
center, memory governance, and prompt trim manifests. The SDK
owns the shape and summary counters, including discovery match counts, memory
hit counts, storage backend counts, context injection counts, ToolCenter counts,
MCP/skill center counts, handoff counts, approval status counts, artifact
counts/bytes, memory governance counts, and prompt-trim presence.
`StorageBackendTrace` deduplicates the backend manifests visible
across those components so a runtime can audit which state lived in core
builtins and which state lived in external PG/vector/graph/object-store
adapters. `ContextMaterialSelectionTrace` summarizes candidate context selection
before injection policy by status, target, score, bytes, and hashes.
`ContextInjectionTrace` summarizes prompt injection decisions by source, target
bucket, status, included count, excluded count, and trimmed count.
`MemoryGovernanceTrace` summarizes memory write allow/rewrite/deny
decisions, risk levels, stores, reasons, and prompt-safe leak hashes. Runtime
code owns where the bundle is stored, how long it is retained, and how it is
queried for product observability or incident review.

`ApprovalTrace` is derived from approval store manifests. It records approval
ids, run/turn ids, status, subject, subject kind, and prompt-safe decision
metadata so eval and replay can assert human-in-loop behavior without importing
identity, notification, ticketing, or workflow runtime code.

`ArtifactTrace` is derived from artifact store manifests. It records artifact
ids, URIs, hashes, content types, byte sizes, artifact kind, tool name, and call
id so eval and replay can assert prompt-unsafe observations were externalized
without embedding artifact bodies in trace summaries.

`HandoffTrace` is derived from target-run prompt metadata. It records
multi-agent handoff status, selected/source sessions, requirements, candidate
counts, and denial/not-found states while product queues, distributed workers,
and orchestration policy remain outside the SDK.

`TraceCorrelationIndex` is generated inside the trace bundle. It gives
provider-neutral cross references across provider calls, tool replay records,
policy decisions, approvals, event entries, and journal replay events by run,
turn, call id, approval id, decision id, and subject. Runtime observability
systems can ingest the index, but the SDK owns the correlation schema.

`TraceReplayHarness`, `TraceReplayComparator`, and `TraceEvalHarness` turn run
trace manifests into deterministic replay steps, baseline diff reports, and
provider-neutral evaluation reports. Replay steps include resume selection,
checkpoint loading, journal events, event-log entries, provider calls, provider
streaming calls using prompt-safe summaries, prompt bucket budget, semantic
prompt trim, global prompt trim, approval records, artifact records, structured
output validation/repair records, handoff records, embedding calls, and lifecycle hook records. The core
checks generic contracts such as final status, iteration limits, provider call
limits, embedding call limits, required embedding providers/models/dimensions,
provider-native tool-call presence/names, required events, required tools,
cost ceilings, event ordering, journal integrity,
resume-plan presence,
resume-plan readiness, expected resume checkpoint ids, handoff status/session constraints, tool execution presence,
tool retry, minimum tool attempt counts, ToolCenter selected mount/tool and
failed-call constraints, required storage backend roles/kinds,
forbidden backend kinds, external-backend limits, event-log presence/types,
terminal events, sequence monotonicity, duplicate sequence limits, lifecycle
hook event/status/failure constraints, required context injection
names/sources/targets/statuses, required included/trimmed/excluded injection
sources, forbidden injection sources/statuses,
trimmed/excluded injection limits, memory governance allow/rewrite/deny and risk
ceilings, prompt bucket budget roles/statuses/over-budget ceilings, runtime
semantic prompt trim roles/statuses/dropped-unit ceilings, approval
status/subject/pending/rejected constraints, artifact kind/tool/content-type
and byte ceilings, and global prompt trim roles/byte ceilings. Runtime code owns
domain-specific eval datasets, baseline selection, scoring policy, dashboards,
and release gates.

`AgentRunStorePort` persists manager-level run state such as queued, running,
cancelling, completed, failed, and interrupted. The SDK ships in-memory, SQLite,
and Markdown stores so lightweight managers can restart and inspect state without
runtime infrastructure. `AgentManagerConcurrencyPolicy` gives the single-process
manager explicit active-run and per-session capacity guards. When
`reject_when_full` is false, the core manager keeps overflow work as queued run
state and starts it after local capacity is released. Unclaimed queued runs can
be cancelled as terminal manager state without cancelling the active session
token or consuming later capacity. Restored queued, running, and cancelling
manager records are marked `interrupted` with `restored_from_status` metadata
because the SDK run store intentionally persists state manifests, not executable
request objects.
`AgentManagerScheduleSnapshot` and `AgentManagerCapacityStatus` expose
provider-neutral schedule/capacity audit manifests so runtimes can inspect
availability, active claims, pending queues, interrupted restored runs, and
capacity rejection reasons without owning core state layout. Production
schedulers, distributed workers, PG-backed workflow state, and tenant isolation
remain runtime or adapter responsibilities.
`AgentLifecycleHookCenter` gives runtimes one SDK-level hook bus for
`run_starting`, `run_completed`, and `run_failed`. Hook records use prompt-safe
task hashes/byte counts and capture hook errors without breaking execution by
default; runtimes can opt into fail-fast hooks when metrics, tenant gates, or
audit systems must be mandatory. Concrete middleware stacks, metrics exporters,
policy services, and UI notifications remain outside `agent_core`.

`HandoffSpec`, `HandoffRequest`, `HandoffRouter`, and `MultiAgentCoordinator`
provide a provider-neutral multi-agent handoff contract. The core can advertise
session capabilities, select targets by tags/tools/skills/priority, run the
selected session through `AgentSessionManager`, and record handoff manifests.
Runtime code owns product queues, cross-process scheduling, retries, UI
orchestration, and domain delegation strategy.

`CapabilityCatalog.discover()` is the SDK-level discovery surface across
actions, local tools, skills, MCP tools, MCP resources, MCP prompts, and MCP
servers. It returns `CapabilityDiscoveryResult` manifests that can be injected
into prompts, stored in traces, or shown by lightweight runtimes without knowing
which subsystem supplied the match. Runtimes still own concrete MCP sessions,
tool credentials, external search indexes, UI filtering, and distributed
capability refresh.

`MCPCenter.refresh_inventory()` provides one auditable refresh pass across MCP
tools, resources, and prompts. It returns per-server
`MCPInventoryRefreshResult` manifests with tool/resource/prompt counts and
partial-failure errors, while `MCPServerState` keeps prompt-safe inventory
counts. Runtime code still decides how MCP servers are launched, authenticated,
isolated, and retried.
`MCPContextMaterialRequest` is the bridge from MCP inventory to prompt shaping:
`MCPCenter.context_materials()` reads selected or ranked resources/prompts and
returns selector-ready `ContextMaterial` records with prompt-safe byte/hash
manifests. This lets MCP participate in the same semantic context
selection/trimming/injection path as memory and runtime hints without turning
MCP process management into SDK code.
Runner integration is explicit: `AgentRunRequest.mcp_context_materials` asks
`AgentRunner` to collect MCP materials and merge them into the existing
`ContextMaterialSelectionRequest`; requests without that field never read MCP
resources/prompts during prompt assembly.

`MCPCenterTrace` and `SkillCenterTrace` are run-level summaries derived from
session/capability manifests. They let replay and eval require MCP server
presence, refresh status, partial-failure limits, loaded skills, and skill
resource view windows without moving MCP process management or skill
distribution into the SDK.

`RunTraceStorePort` gives the bundle a persistence boundary. The SDK ships
in-memory, SQLite, and Markdown stores for lightweight use, while production
PG/object-storage/observability integrations should live in runtime or adapter
packages.

`ArtifactStorePort` keeps oversized observations out of prompt text while
preserving retrievable content. The SDK ships in-memory, SQLite, and Markdown
stores for local runs. Production filesystem/object-storage integrations remain
runtime responsibilities.

`LLMProviderCenter` records provider-neutral call manifests for completed and
failed attempts. The core records provider name, model, attempt, streamed flag,
usage, retryability, routed request shape, and original request shape. Concrete
provider clients, credentials, rate limits, and vendor-specific response
payloads remain runtime-owned.
`LLMModelCapabilities` lets runtimes declare context windows, output limits,
streaming support, tool-call support, JSON mode, structured-output support, and
modalities without importing a vendor SDK. The center uses those declarations to
skip incompatible routes when a request declares required capabilities or
estimated token/output size. Routed call manifests retain the selected
capability profile, and trace evals can require specific provider names, model
names, or capabilities such as `structured_output`, `json_mode`, `tool_calls`,
and `streaming`.
`LLMProviderRoutePlan` is the preflight form of the same decision. It lists each
candidate provider, selected route, fallback candidates, capability/model
rejection reasons, requested provider/model, and streamed mode without invoking
a model. Provider call records attach the route plan manifest so traces can
explain why a runtime chose or skipped a provider while concrete credentials,
tenant routing, and vendor clients stay outside the SDK.
`LLMToolContract`, `LLMToolChoice`, and `LLMResponseFormat` are request-level
contracts for native model tools and constrained output. They deliberately stop
at the provider-neutral shape: OpenAI-compatible tools/response formats,
Anthropic tool use, Gemini function declarations, local-model JSON schemas, and
gateway payload details remain adapter responsibilities. `LLMModelCapabilities`
can reject routes that cannot satisfy these contracts before a provider call is
attempted.
`LLMToolCall` is the matching response-level contract. A provider adapter can
decode native tool calls into `LLMResponse.tool_calls` or stream `tool_call`
events. When `ReActConfig.native_tool_calls` is enabled, the executor runs those
calls through the same policy, approval, replay, retry, and tool-result
compaction boundaries used by JSON ReAct actions, then appends provider-neutral
`role=tool` messages for the next turn. This keeps OpenAI-compatible tool calls,
Anthropic tool use, and local model function calling behind the same core loop.
`AgentSession.native_tool_calls` and `AgentRunRequest.native_tool_calls` expose
that mode at the runner boundary so consuming runtimes can enable it by default
or roll it out per request without bypassing `AgentRunner`.
`LLMContentPart` extends `LLMMessage` beyond a single text field while keeping
the same boundary. The core can represent text, JSON, image, audio, file, and
binary parts and route them through declared `modalities`; runtimes still own
file reads, object storage, upload handles, signed URLs, and vendor-specific
message part payloads.
`TransportLLMProvider`, `LLMTransportPort`, and `LLMProviderCodecPort` form the
dependency-free adapter boundary: runtimes can provide a transport and optional
vendor codec while the SDK keeps provider-neutral request, response, stream, and
error/retry semantics.
`OpenAICompatibleLLMProviderCodec` is a built-in codec for Chat
Completions-style payloads. It maps SDK messages, multimodal parts, native tool
contracts, tool choices, response formats, usage, tool calls, and basic stream
chunks without depending on the OpenAI SDK; HTTP/auth/model deployment remain
runtime responsibilities.
`LLMStreamAccumulator` gives streaming calls a single reconstruction and audit
contract: events become an `LLMResponse`, while manifests retain event counts,
event types, byte counts, usage, finish reason, and action presence without
embedding raw streamed content.
`LLMRetryPolicy` and `LLMUsageLimits` give the SDK a provider-neutral way to
describe retry/fallback behavior, call-attempt ceilings, token ceilings, and
cost ceilings. Runtime code can derive these policies from tenants, tasks, or
model classes, while real rate limiters and vendor quotas stay outside core.
For streaming calls, core treats provider `error` events as failed attempts,
records streamed failure metadata, and applies retry/fallback and budget checks
before yielding a successful stream to callers. Completed streamed calls include
the standard stream summary in provider call metadata. Trace evals can require
streaming calls, required or forbidden stream event types, and maximum stream
error counts while keeping streamed content out of the trace.

`EmbeddingProviderPort` and `EmbeddingProviderCenter` give semantic retrieval a
separate provider-neutral boundary from chat/completion models. The core records
embedding request, response, route, provider spec, and call manifests without
depending on OpenAI, Gemini, local model, or vector-service clients.
`DeterministicEmbeddingProvider` exists only for tests and lightweight local
ranking. Production embedding clients, batch policy, vector-store upserts,
tenant credentials, and vendor rate limits remain runtime or adapter
responsibilities. Run traces can include the embedding center manifest so
`TraceReplayHarness` and `DefaultTraceEvaluator` can audit semantic retrieval
calls without storing raw embedded text or vectors.

`ToolReplayStorePort` gives tool replay the same port-based shape as memory and
journals. Core replay manifests include replay keys, invocation argument hashes,
and prompt-safe result manifests. Runtime code owns durable storage, retention,
tenant isolation, and whether replay can cross a run boundary.
The SDK ships in-memory, SQLite, and Markdown stores for local or inspectable
replay. Production PG/object-storage backends should live in the runtime or a
separate adapter package.

`ToolCenter` aggregates local, MCP-backed, or runtime-owned tool runtimes behind
one route boundary. `ToolRoutePlan` explains which mount and concrete tool name
will handle a requested name or alias, including disabled/unknown candidates
when the mounted runtime exposes them. `ToolCenterCallRecord` records the
selected route and prompt-safe result manifest after invocation, while concrete
tool side effects and runtime process/network isolation remain outside the SDK.
`ToolCenterTrace` lifts those route/call records into run traces so eval suites
can require selected mounts, selected concrete tools, requested names, ready
route plans, and failed-call ceilings without understanding the runtime.

`ToolExecutionCenter` wraps any `ToolRuntimePort` with provider-neutral retry
and audit semantics. `ToolRetryPolicy` decides whether retryable failed results
or exceptions may be attempted again, and `ToolExecutionRecord` captures attempt
status, retryability, final result, schema validation, and summary metadata.
Before dispatching to the runtime, core validates `ToolSpec.parameters_schema`
when a matching spec is available; invalid arguments become a failed result with
a `SchemaValidationResult` manifest and do not trigger tool side effects.
`ReActExecutor` records this summary on tool results and `tool_finished` events.
Runtime code still owns concrete tool implementations, side-effect safety,
idempotency rules, and distributed retry scheduling.

`MemoryPort` is the execution dependency for long-term recall, while
`MemoryCenter` carries richer backend contracts through `MemoryStoreSpec`,
`MemoryQuery`, `MemoryRoute`, and `MemorySearchPlan`. Query modes cover keyword,
semantic, vector, graph, and hybrid retrieval; store specs advertise backend
kind, namespace coverage, tags, location, runtime ownership, and vector/graph
capability flags. `ExternalMemoryStore` wraps runtime-owned PG, vector DB,
graph/RAG, or product-memory adapters with the same manifest shape as SDK-local
stores without importing their drivers. This lets a runtime mount external
memory adapters without changing runner/ReAct code, while the SDK still records
a prompt-safe route and query plan for trace/replay.
External memory adapters also emit `ExternalMemoryCallRecord` manifests for
search and write calls. These records capture backend kind, status, errors,
hit counts, query manifests, and write content hashes/byte counts without
requiring a Postgres, vector DB, or graph driver inside `agent_core`.

`ContextMaterialStorePort` is the matching candidate-context boundary for
semantic prompt shaping. `ContextMaterialCenter` can route queries across SDK
stores and runtime-owned SQLite, Markdown, PG, vector, graph, or product API
stores through `ContextMaterialStoreSpec`, `ContextMaterialQuery`,
`ContextMaterialRoute`, and `ContextMaterialSearchPlan`. The SDK ships an
in-memory store and `ExternalContextMaterialStore` for prompt-safe call audit;
concrete durable drivers remain outside core. `AgentRunRequest.context_material_query`
lets `AgentRunner` collect those candidates explicitly and pass them through
the existing selector/trimming/injection pipeline.

`ReActExecutor` emits monotonic `AgentEvent.sequence` values. Lightweight users
can capture them with `ListEventSink`; inspectable or local durable runs can use
`SQLiteEventSink` or `MarkdownEventSink`. Production runtimes should implement
`EventSinkPort` or `EventLogPort` for their own logs, UI streams, metrics,
retention policy, or audit systems. `EventStreamCursor` and `EventStreamBatch`
give those runtimes a provider-neutral paging contract with run filtering,
sequence cursors, event-type filters, `has_more`, and terminal-run detection;
the SDK does not own the SSE/WebSocket transport.

`ContextReducerPort` handles automatic timeline/context reduction as a core
mechanic. `DefaultContextReducer` is deterministic and dependency-free, while
production runtimes can implement the same port with model-backed summaries,
archive storage, vector indexes, or graph memory. The reducer returns manifests
so compression can be audited and replayed.

`AgentSession.context_reducer` lets the runner apply that reducer as part of the
core execution path. When the active timeline exceeds
`RuntimeBudget.max_timeline_bytes`, `AgentRunner` reduces the timeline before
prompt assembly, updates the `TimelineStore` with compressed head text and
archive refs, and attaches the reduction manifest to both `AgentRunOutcome` and
prompt metadata.

`PlannerPort` is a core contract because plan state, dependency readiness,
updates, and manifests are reusable across agent domains. The core ships
`InMemoryPlanner`, `PersistentPlanner`, lightweight in-memory/SQLite/Markdown
planner stores, and `PlanExecutor` for sequential ready-step execution through
`AgentSessionManager`. `PlannerTrace` turns planner/executor manifests into
replayable and evaluable plan ids, step statuses, execution statuses, and
failure/blockage counts. Raven-specific decomposition, code repair plans, approval
flows, distributed scheduling, PG-backed workflow state, and product planner
audit policy should live in a runtime or adapter package that implements the
same ports.

`ApprovalStorePort` is a core contract for human-in-loop pauses. Policies can
produce `ApprovalRequest` manifests; ReAct execution records them as
`ApprovalRecord` entries and emits `approval_requested` events. Runtime code owns
the approval UI, identity checks, notifications, and durable workflow backend.
The SDK ships Null, in-memory, SQLite, and Markdown stores so lightweight runs
can persist approval state without adopting a product database. Production
approval services, ticketing systems, and RBAC should live in the runtime or a
separate adapter package.

`ApprovalCenter` is the SDK-facing approval operation facade. It can inspect
queues through `ApprovalQueueView`, apply approve/reject/cancel decisions as
`ApprovalResolution` manifests, and build `ApprovalResumeContext` from resolved
records. Runtime code still owns who can decide, how approvals are routed, and
which external workflow system stores or mirrors those decisions.

`ApprovalResumeContext` is the core continuation contract. A runtime can resolve
an approval through any backend, convert the resulting `ApprovalRecord` into a
grant, and pass that context into `AgentRunRequest`. `ReActExecutor` only uses
approved grants whose subject matches the current policy gate, then emits
`approval_resumed`. This keeps approval-aware resume, replay, and policy
semantics in the SDK while leaving people, permissions, and workflow routing to
the runtime.

`RunInterrupt` is the SDK-level contract for cancelled, timed-out, deadline, and
external-interrupt runs. `CancelToken` keeps a prompt-safe interrupt manifest,
`AgentRunRequest.timeout_seconds` applies a request-level deadline, and
`ReActExecutor` records timeout checkpoints plus `run_timeout` events at
provider/tool await boundaries before finishing the run with status `timeout`.
Runtimes own process killing, worker cancellation, UI buttons, queue state, and
distributed lease cleanup; the SDK owns the portable lifecycle semantics,
checkpoint shape, event schema, and trace/replay metadata.

## Adapter Policy

Adapters are useful, but they should not live in this repository while the SDK
base is being stabilized. Build runtime adapters in the consuming runtime or in
separate adapter repositories. The core package must remain importable without
OpenAI Agents SDK, RavenStorm, FastAPI, Graphiti, Redis, SQLAlchemy, or MCP SDK.

## Yaklang-Inspired Core Ideas

`agent_core` keeps the generally useful parts of Yaklang-style agent systems:

- semantic prompt partitioning;
- automatic trimming;
- context injection;
- memory-backed continuity;
- tool, skill, and MCP capability orchestration.

It does not copy or translate Yaklang source code.

## Prompt Boundary

The SDK owns prompt mechanics, not business wording. `PromptIR` records semantic
buckets, stable ordering, cache hints, hashes, and trim metadata. Runtimes provide
the actual instructions, task contracts, examples, and domain language.

`AgentRunner` applies `PromptIR.trim_to_budget()` with
`RuntimeBudget.max_prompt_bytes` before provider calls, so automatic trimming is
part of the core execution path rather than a Raven-specific adapter behavior.
If `AgentSession.context_reducer` is configured, the runner also applies
timeline reduction before prompt assembly using
`RuntimeBudget.max_timeline_bytes`.

`PromptTrimPlan`, `PromptTrimRule`, `PromptTrimStep`, and `PromptTrimResult`
make prompt trimming auditable before and after the cut. The default rules trim
volatile timeline material first, then semi-dynamic recall/schema buckets, then
capability inventory, while protecting high-static system rules and preserving a
minimum dynamic task window. Runtimes can provide custom rules for code, ops, or
security agents without changing bucket order or provider calls. Trace evals
can require or forbid global prompt trim, require specific trimmed semantic
roles, and cap original/final prompt bytes.

`PromptBucketBudgetPolicy` adds an earlier bucket-local budget pass. It can cap
specific semantic buckets such as memory, skills, capability inventory, or open
timeline context, while marking high-static or current-task buckets as
protected. `AgentSession.prompt_bucket_budget_policy` lets the runner apply this
before the global prompt trim. The SDK records the policy, per-bucket decisions,
trimmed count, protected count, and any over-budget buckets in prompt and trace
manifests. Runtimes own the actual budget numbers and rollout policy.

`PromptSemanticReducerPort` is the runtime-time semantic trimming boundary for
already assembled `PromptIR`. `AgentRunner` calls
`AgentSession.prompt_semantic_reducer` after bucket budgets and before the final
byte trim. The built-in `DefaultPromptSemanticReducer` is deterministic and
query-overlap based for tests and lightweight agents; Raven, code agents, or
ops agents can replace it with embedding-backed, LLM-backed, or domain-specific
reducers. `PromptSemanticTrimRequest` and `PromptSemanticTrimResult` keep the
audit provider-neutral by recording hashes, roles, per-bucket decisions,
selected/dropped unit counts, and convergence without introducing a concrete
semantic model into core.

`ContextInjection` is the SDK-level insertion record for resumable state, memory
continuity, operator hints, and runtime-supplied context. It declares the target
bucket, source, priority, and metadata. The runtime still owns the actual content
and policy for adding it. Trace evals can require specific injection names,
sources, target buckets, statuses, included sources, trimmed sources, and
excluded sources, giving Yaklang-style context routing and auto-trimming a
provider-neutral acceptance contract.

`ContextMaterial` and `ContextMaterialSelectorPort` sit one step earlier than
injection. Runtimes can gather candidate material from local memory, Markdown,
PG, vector search, graph memory, tool caches, or product APIs, then hand those
candidates to the SDK as provider-neutral material records. The built-in
`DefaultContextMaterialSelector` ranks candidates by deterministic task-term
overlap, role mapping, priority, max material count, and byte budget, returning
`ContextInjection` objects plus a manifest of selected and dropped candidates.
The manifest stores names, roles, targets, scores, byte counts, and hashes rather
than raw hidden context bodies.
`AgentSession.context_material_selector` lets `AgentRunner` apply this step
before prompt assembly when a request supplies context materials or an explicit
selection request.
`AgentRunTraceBundle`, `TraceReplayHarness`, and `DefaultTraceEvaluator` expose
that manifest as a first-class trace contract, so suites can require selected
materials, forbid drop statuses, cap dropped count, or cap selected bytes.

`ContextInjectionPolicy` is the SDK-level guardrail applied before prompt
assembly. It can restrict target buckets, trim each injected block, cap total
injection bytes, and emit per-injection decisions into the prompt manifest. This
keeps resume, memory, approval, and runtime-supplied context on one auditable
path without adding Raven-specific prompt content to core.

When memory is enabled on an `AgentSession`, `AgentRunner` recalls memory through
`MemoryPort` and injects the rendered hits as `ContextInjection(source="memory")`.
Standalone `ReActExecutor` still supports direct memory messages for tests and
small embeddings that do not use the full runner. `InMemoryMemoryStore` can
optionally use an `EmbeddingProviderPort` for semantic and hybrid ranking, while
SQLite and Markdown remain dependency-free keyword stores.

`MemorySearchPlan` is intentionally separate from the storage implementation.
It proves which stores were selected, which route filters were consumed by the
center, and which backend filters were passed through. Concrete PG/vector/graph
stores can translate the same `MemoryQuery` fields into SQL, vector search,
graph traversal, or product APIs outside this repository.

`RuleBasedMemoryGovernance` is the built-in dependency-free write gate. It can
deny empty or unsafe writes, reject protected-scope target or secret leaks,
enforce allowed global buckets, truncate over-budget content, and record
prompt-safe decision manifests. Runtimes can replace the governance port with
tenant-specific policy while keeping trace and eval contracts stable.
