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
| LLM provider center | Concrete model clients, credentials, rate limits |
| Tool center | Real tools, shell/file/network access, sandboxing |
| MCP center | MCP server deployment, credentials, process lifecycle |
| Memory center and `MemoryPort` | Graphiti, RAG, PG/vector/graph/product memory adapters |
| Harness journal, replay, and `AgentJournalStorePort` | PG/event-log/workflow persistence adapters |
| Prompt buckets, trimming, and injection | Domain-specific prompt material and task contracts |
| Harness/ReAct runner | UI events, API routes, production persistence backend |
| Sequenced event stream | UI rendering, logs, metrics, audit pipeline |
| Policy ports | Operator approval UX and organization policy |

## Data Backend Boundary

Core ships lightweight implementations so the SDK can run by itself:

| Data area | Port | Built-in implementations | Runtime implementations |
| --- | --- | --- | --- |
| Memory | `MemoryPort` | In-memory, SQLite, Markdown | PG, vector DB, graph/RAG, product knowledge |
| Harness journal | `AgentJournalStorePort` | In-memory snapshot store, SQLite snapshot store, Markdown snapshot store | PG, object storage, workflow DB, audit event log |
| Artifacts | `ArtifactStorePort` | In-memory artifact store | Filesystem, object storage, CI/build artifacts |
| Events | `EventSinkPort` | Protocol only | UI stream, logs, metrics, audit pipeline |

The SDK must not require a production database driver. Production backends should
be added by the host runtime or by separate adapter packages that implement the
same ports.

`AgentJournalReplay` turns a journal snapshot into a replayable event manifest
and reports consistency issues before a runtime depends on that state for UI,
debugging, audit, or resume decisions.

`ReActExecutor` emits monotonic `AgentEvent.sequence` values. Lightweight users
can capture them with `ListEventSink`; production runtimes should implement
`EventSinkPort` for their own logs, UI streams, metrics, or audit systems.

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

`ContextInjection` is the SDK-level insertion record for resumable state, memory
continuity, operator hints, and runtime-supplied context. It declares the target
bucket, source, priority, and metadata. The runtime still owns the actual content
and policy for adding it.

When memory is enabled on an `AgentSession`, `AgentRunner` recalls memory through
`MemoryPort` and injects the rendered hits as `ContextInjection(source="memory")`.
Standalone `ReActExecutor` still supports direct memory messages for tests and
small embeddings that do not use the full runner.
