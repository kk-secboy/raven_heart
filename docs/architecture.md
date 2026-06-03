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

`AgentJournalReplay` turns a journal snapshot into a replayable event manifest
and reports consistency issues before a runtime depends on that state for UI,
debugging, audit, or resume decisions.

`ResumeCandidate` and `ResumeIndex` turn the latest checkpoint per run into a
standard manifest with a `ResumeToken`, checkpoint state, terminal status, and
run metadata. `ResumePlan` adds the preflight decision layer: selected
checkpoint, ready/unavailable status, terminal-run warnings, missing-candidate
errors, and a small summary manifest that can be persisted with manager run
state. `AgentRunner.resume()` and `AgentSessionManager.resume()` convert those
plans into `AgentRunRequest` objects without runtime-side token assembly. The
SDK owns discovery, serialization, local resume entrypoints, and preflight
status; runtime code owns worker selection, user-facing recovery flows, and
distributed resume scheduling.

`AgentRunTraceBundle` aggregates per-run session, prompt, journal replay,
provider audit, tool replay, approvals, event log, resume, timeline reduction,
capability discovery, memory recall/search, memory governance, and prompt trim manifests. The SDK
owns the shape and summary counters, including discovery match counts, memory
hit counts, storage backend counts, context injection counts, memory governance
counts, and prompt-trim presence. `StorageBackendTrace` deduplicates the backend manifests visible
across those components so a runtime can audit which state lived in core
builtins and which state lived in external PG/vector/graph/object-store
adapters. `ContextInjectionTrace` summarizes prompt injection decisions by
source, target bucket, status, included count, excluded count, and trimmed
count. `MemoryGovernanceTrace` summarizes memory write allow/rewrite/deny
decisions, risk levels, stores, reasons, and prompt-safe leak hashes. Runtime
code owns where the bundle is stored, how long it is retained, and how it is
queried for product observability or incident review.

`TraceCorrelationIndex` is generated inside the trace bundle. It gives
provider-neutral cross references across provider calls, tool replay records,
policy decisions, approvals, event entries, and journal replay events by run,
turn, call id, approval id, decision id, and subject. Runtime observability
systems can ingest the index, but the SDK owns the correlation schema.

`TraceReplayHarness`, `TraceReplayComparator`, and `TraceEvalHarness` turn run
trace manifests into deterministic replay steps, baseline diff reports, and
provider-neutral evaluation reports. The core checks generic contracts such as
final status, iteration limits, provider call limits, required events, required
tools, cost ceilings, event ordering, journal integrity, resume-plan presence,
resume-plan readiness, expected resume checkpoint ids, tool execution presence,
tool retry, minimum tool attempt counts, required storage backend roles/kinds,
forbidden backend kinds, external-backend limits, required context injection
sources/targets, forbidden injection sources, and trimmed/excluded injection
limits, memory governance allow/rewrite/deny and risk ceilings, and prompt
bucket budget roles/statuses/over-budget ceilings. Runtime code owns
domain-specific eval datasets, baseline selection, scoring policy, dashboards,
and release gates.

`AgentRunStorePort` persists manager-level run state such as queued, running,
cancelling, completed, failed, and interrupted. The SDK ships in-memory, SQLite,
and Markdown stores so lightweight managers can restart and inspect state without
runtime infrastructure. `AgentManagerConcurrencyPolicy` gives the single-process
manager explicit active-run and per-session capacity guards.
`AgentManagerScheduleSnapshot` and `AgentManagerCapacityStatus` expose
provider-neutral schedule/capacity audit manifests so runtimes can inspect
availability, active claims, interrupted restored runs, and capacity rejection
reasons without owning core state layout. Production schedulers, distributed
workers, PG-backed workflow state, and tenant isolation remain runtime or
adapter responsibilities.

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
`TransportLLMProvider`, `LLMTransportPort`, and `LLMProviderCodecPort` form the
dependency-free adapter boundary: runtimes can provide a transport and optional
vendor codec while the SDK keeps provider-neutral request, response, stream, and
error/retry semantics.
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
the standard stream summary in provider call metadata.

`ToolReplayStorePort` gives tool replay the same port-based shape as memory and
journals. Core replay manifests include replay keys, invocation argument hashes,
and prompt-safe result manifests. Runtime code owns durable storage, retention,
tenant isolation, and whether replay can cross a run boundary.
The SDK ships in-memory, SQLite, and Markdown stores for local or inspectable
replay. Production PG/object-storage backends should live in the runtime or a
separate adapter package.

`ToolExecutionCenter` wraps any `ToolRuntimePort` with provider-neutral retry
and audit semantics. `ToolRetryPolicy` decides whether retryable failed results
or exceptions may be attempted again, and `ToolExecutionRecord` captures attempt
status, retryability, final result, and summary metadata. `ReActExecutor`
records this summary on tool results and `tool_finished` events. Runtime code
still owns concrete tool implementations, side-effect safety, idempotency rules,
and distributed retry scheduling.

`MemoryPort` is the execution dependency for long-term recall, while
`MemoryCenter` carries richer backend contracts through `MemoryStoreSpec`,
`MemoryQuery`, `MemoryRoute`, and `MemorySearchPlan`. Query modes cover keyword,
semantic, vector, graph, and hybrid retrieval; store specs advertise backend
kind, namespace coverage, tags, and vector/graph capability flags. This lets a
runtime mount PG, vector DB, graph/RAG, or product-memory adapters without
changing runner/ReAct code, while the SDK still records a prompt-safe route and
query plan for trace/replay.

`ReActExecutor` emits monotonic `AgentEvent.sequence` values. Lightweight users
can capture them with `ListEventSink`; inspectable or local durable runs can use
`SQLiteEventSink` or `MarkdownEventSink`. Production runtimes should implement
`EventSinkPort` or `EventLogPort` for their own logs, UI streams, metrics,
retention policy, or audit systems.

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
`AgentSessionManager`. Raven-specific decomposition, code repair plans, approval
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
security agents without changing bucket order or provider calls.

`PromptBucketBudgetPolicy` adds an earlier bucket-local budget pass. It can cap
specific semantic buckets such as memory, skills, capability inventory, or open
timeline context, while marking high-static or current-task buckets as
protected. `AgentSession.prompt_bucket_budget_policy` lets the runner apply this
before the global prompt trim. The SDK records the policy, per-bucket decisions,
trimmed count, protected count, and any over-budget buckets in prompt and trace
manifests. Runtimes own the actual budget numbers and rollout policy.

`ContextInjection` is the SDK-level insertion record for resumable state, memory
continuity, operator hints, and runtime-supplied context. It declares the target
bucket, source, priority, and metadata. The runtime still owns the actual content
and policy for adding it.

`ContextInjectionPolicy` is the SDK-level guardrail applied before prompt
assembly. It can restrict target buckets, trim each injected block, cap total
injection bytes, and emit per-injection decisions into the prompt manifest. This
keeps resume, memory, approval, and runtime-supplied context on one auditable
path without adding Raven-specific prompt content to core.

When memory is enabled on an `AgentSession`, `AgentRunner` recalls memory through
`MemoryPort` and injects the rendered hits as `ContextInjection(source="memory")`.
Standalone `ReActExecutor` still supports direct memory messages for tests and
small embeddings that do not use the full runner.

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
