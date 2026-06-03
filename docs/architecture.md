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
| Prompt buckets, trimming, and injection | Domain-specific prompt material and task contracts |
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
| Tool replay | `ToolReplayStorePort` | In-memory replay store | PG, SQLite, object storage, workflow replay DB |
| Artifacts | `ArtifactStorePort` | In-memory artifact store | Filesystem, object storage, CI/build artifacts |
| Approvals | `ApprovalStorePort` | Null store, in-memory approval queue | Approval service, ticketing/workflow DB, operator UI |
| Events | `EventSinkPort` | Protocol only | UI stream, logs, metrics, audit pipeline |

The SDK must not require a production database driver. Production backends should
be added by the host runtime or by separate adapter packages that implement the
same ports.

`AgentJournalReplay` turns a journal snapshot into a replayable event manifest
and reports consistency issues before a runtime depends on that state for UI,
debugging, audit, or resume decisions.

`LLMProviderCenter` records provider-neutral call manifests for completed and
failed attempts. The core records provider name, model, attempt, streamed flag,
usage, retryability, and request shape. Concrete provider clients, credentials,
rate limits, and vendor-specific response payloads remain runtime-owned.

`ToolReplayStorePort` gives tool replay the same port-based shape as memory and
journals. Core replay manifests include replay keys, invocation argument hashes,
and prompt-safe result manifests. Runtime code owns durable storage, retention,
tenant isolation, and whether replay can cross a run boundary.

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
