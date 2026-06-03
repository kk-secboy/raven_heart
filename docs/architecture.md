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
| LLM provider center and call audit | Concrete model clients, credentials, rate limits |
| Tool center | Real tools, shell/file/network access, sandboxing |
| Tool replay records and store port | Durable replay backend, retention, cross-run replay policy |
| MCP center | MCP server deployment, credentials, process lifecycle |
| Memory center and `MemoryPort` | Graphiti, RAG, PG/vector/graph/product memory adapters |
| Harness journal, replay, and `AgentJournalStorePort` | PG/event-log/workflow persistence adapters |
| Run trace bundle | Durable trace export, observability pipeline, retention |
| Trace replay/eval harness | Domain eval suites, dashboards, regression policy |
| Manager run state and `AgentRunStorePort` | Product workflow DB, scheduling, distributed workers |
| Multi-agent handoff protocol | Product queues, distributed workers, business orchestration |
| Prompt buckets, trimming, and injection | Domain-specific prompt material and task contracts |
| Structured output specs and validation lifecycle | Domain schemas, typed business objects, persistence |
| Context reducer protocol and reduction manifests | Domain summarizers, archive stores, retrieval policy |
| Planner protocol and plan state | Domain-specific plan generation and workflow policy |
| Harness/ReAct runner | UI events, API routes, production persistence backend |
| Sequenced event stream | UI rendering, logs, metrics, audit pipeline |
| Policy ports | Operator approval UX and organization policy |
| Approval store and approval manifests | Identity, approval UI, workflow routing, escalation policy |

## Data Backend Boundary

Core ships lightweight implementations so the SDK can run by itself:

| Data area | Port | Built-in implementations | Runtime implementations |
| --- | --- | --- | --- |
| Memory | `MemoryPort` | In-memory, SQLite, Markdown | PG, vector DB, graph/RAG, product knowledge |
| Harness journal | `AgentJournalStorePort` | In-memory snapshot store, SQLite snapshot store, Markdown snapshot store | PG, object storage, workflow DB, audit event log |
| Tool replay | `ToolReplayStorePort` | In-memory, SQLite, Markdown | PG, object storage, workflow replay DB |
| Run traces | `RunTraceStorePort` | In-memory, SQLite, Markdown | PG, object storage, observability pipeline |
| Manager runs | `AgentRunStorePort` | In-memory, SQLite, Markdown | PG, workflow DB, scheduler state |
| Artifacts | `ArtifactStorePort` | In-memory, SQLite, Markdown | Filesystem, object storage, CI/build artifacts |
| Approvals | `ApprovalStorePort` | Null, in-memory, SQLite, Markdown | Approval service, ticketing/workflow DB, operator UI |
| Events | `EventSinkPort` | Protocol only | UI stream, logs, metrics, audit pipeline |

Structured output is intentionally contract-based rather than provider-specific.
`StructuredOutputSpec` can be attached to an `AgentRunRequest`; the runner injects
the schema into prompt context, and `ReActExecutor` validates `finish.output`.
Invalid output becomes repair feedback inside the ReAct loop. Runtime code owns
the domain schema, typed object mapping, and downstream storage.

The SDK must not require a production database driver. Production backends should
be added by the host runtime or by separate adapter packages that implement the
same ports.

`AgentJournalReplay` turns a journal snapshot into a replayable event manifest
and reports consistency issues before a runtime depends on that state for UI,
debugging, audit, or resume decisions.

`AgentRunTraceBundle` aggregates per-run session, prompt, journal replay,
provider audit, tool replay, approvals, event log, resume, and timeline reduction
manifests. The SDK owns the shape and summary counters. Runtime code owns where
the bundle is stored, how long it is retained, and how it is queried for product
observability or incident review.

`TraceReplayHarness` and `TraceEvalHarness` turn run trace manifests into
deterministic replay steps and provider-neutral evaluation reports. The core
checks generic contracts such as final status, iteration limits, provider call
limits, required events, required tools, cost ceilings, and journal integrity.
Runtime code owns domain-specific eval datasets, scoring policy, dashboards, and
release gates.

`AgentRunStorePort` persists manager-level run state such as queued, running,
cancelling, completed, failed, and interrupted. The SDK ships in-memory, SQLite,
and Markdown stores so lightweight managers can restart and inspect state without
runtime infrastructure. Production schedulers, distributed workers, PG-backed
workflow state, and tenant isolation remain runtime or adapter responsibilities.

`HandoffSpec`, `HandoffRequest`, `HandoffRouter`, and `MultiAgentCoordinator`
provide a provider-neutral multi-agent handoff contract. The core can advertise
session capabilities, select targets by tags/tools/skills/priority, run the
selected session through `AgentSessionManager`, and record handoff manifests.
Runtime code owns product queues, cross-process scheduling, retries, UI
orchestration, and domain delegation strategy.

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
usage, retryability, and request shape. Concrete provider clients, credentials,
rate limits, and vendor-specific response payloads remain runtime-owned.
For streaming calls, core treats provider `error` events as failed attempts,
records streamed failure metadata, and applies retry/fallback and budget checks
before yielding a successful stream to callers.

`ToolReplayStorePort` gives tool replay the same port-based shape as memory and
journals. Core replay manifests include replay keys, invocation argument hashes,
and prompt-safe result manifests. Runtime code owns durable storage, retention,
tenant isolation, and whether replay can cross a run boundary.
The SDK ships in-memory, SQLite, and Markdown stores for local or inspectable
replay. Production PG/object-storage backends should live in the runtime or a
separate adapter package.

`ReActExecutor` emits monotonic `AgentEvent.sequence` values. Lightweight users
can capture them with `ListEventSink`; production runtimes should implement
`EventSinkPort` for their own logs, UI streams, metrics, or audit systems.

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
updates, and manifests are reusable across agent domains. The core only ships a
lightweight `InMemoryPlanner`; Raven-specific decomposition, code repair plans,
approval flows, and durable planner storage should live in a runtime or adapter
package that implements the same port.

`ApprovalStorePort` is a core contract for human-in-loop pauses. Policies can
produce `ApprovalRequest` manifests; ReAct execution records them as
`ApprovalRecord` entries and emits `approval_requested` events. Runtime code owns
the approval UI, identity checks, notifications, and durable workflow backend.
The SDK ships Null, in-memory, SQLite, and Markdown stores so lightweight runs
can persist approval state without adopting a product database. Production
approval services, ticketing systems, and RBAC should live in the runtime or a
separate adapter package.

`ApprovalResumeContext` is the core continuation contract. A runtime can resolve
an approval through any backend, convert the resulting `ApprovalRecord` into a
grant, and pass that context into `AgentRunRequest`. `ReActExecutor` only uses
approved grants whose subject matches the current policy gate, then emits
`approval_resumed`. This keeps approval-aware resume, replay, and policy
semantics in the SDK while leaving people, permissions, and workflow routing to
the runtime.

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

`ContextInjection` is the SDK-level insertion record for resumable state, memory
continuity, operator hints, and runtime-supplied context. It declares the target
bucket, source, priority, and metadata. The runtime still owns the actual content
and policy for adding it.

When memory is enabled on an `AgentSession`, `AgentRunner` recalls memory through
`MemoryPort` and injects the rendered hits as `ContextInjection(source="memory")`.
Standalone `ReActExecutor` still supports direct memory messages for tests and
small embeddings that do not use the full runner.
