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
- Provider call records and usage/failure manifests.
- Tool registry and tool center.
- Tool replay records and replay store port.
- Skill center and skill context injection.
- MCP center and SDK-free stdio connector.
- Prompt buckets and context trimming.
- Provider-neutral prompt IR with semantic bucket trimming.
- Context reducer port and deterministic timeline reduction.
- Runner-level automatic timeline reduction before prompt assembly.
- Context injection records for resume, memory, runtime hints, and other bucketed material.
- SQLite, Markdown, and in-memory stores for lightweight memory.
- Pluggable journal stores for harness checkpoint/resume persistence.
- Journal replay manifests and snapshot consistency audit.
- Run trace bundle for provider/tool/approval/journal/event audit aggregation.
- Run trace store port plus in-memory, SQLite, and Markdown trace stores.
- Trace replay and evaluation harness for provider-neutral run audits.
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
- `AgentRunRequest.resume_token` for injecting checkpoint state into the next run.
- `AgentRunner.resume()`, `AgentSessionManager.resume()`, and
  `AgentSessionManager.start_resume()` for SDK-level resume entrypoints.
- Run manifest and error recording.
- Terminal status checks.
- Policy terminal statuses: `denied` and `approval_required`.
- `AgentJournalReplay` for replayable event timelines and journal consistency audit.
- `AgentRunTraceBundle` for per-run prompt, provider, approval, replay,
  journal, and event manifests.
- `AgentRunStorePort` for manager-level queued/running/completed state.
- `InMemoryAgentRunStore`, `SQLiteAgentRunStore`, and `MarkdownAgentRunStore`
  for lightweight run state persistence.
- `AgentManagerConcurrencyPolicy` for single-process active-run capacity limits.

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

- `StructuredOutputSpec` defines a request-level final answer contract.
- `JsonStructuredOutputValidator` validates a deterministic JSON schema subset.
- `StructuredOutputValidatorPort` lets runtimes replace validation without
  changing ReAct execution.
- `AgentRunRequest.structured_output` injects the schema into prompt context.
- `ReActExecutor` validates `finish.output`; if invalid, it feeds a repair
  message back into the loop before failing the run.

The SDK owns schema injection, validation lifecycle, repair feedback, and
manifests. Runtimes own domain schemas, typed business objects, UI rendering,
and downstream persistence.

### Events

- `AgentEvent` and `EventSinkPort`.
- Monotonic event sequencing inside `ReActExecutor`.
- `EventLogPort` for event logs that can return stored records and manifests.
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
  resume, and timeline reduction manifests when available.
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

- `TraceReplayHarness` builds a deterministic replay timeline from journal and
  event-log trace manifests.
- `TraceReplayComparator` compares two trace manifests and reports deterministic
  replay diffs for regression baselines.
- `TraceEvalSpec` defines provider-neutral expectations such as status,
  iteration limits, provider call limits, required events, and required tools.
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

### LLM Providers

- `LLMProviderPort` protocol.
- Provider registry and routing.
- `LLMRetryPolicy` for retry/fallback behavior.
- `LLMUsageLimits` for cost, call-attempt, and token budgets.
- Usage/failure accounting.
- Budget checks.
- Request, response, stream event, route, and call record manifests.
- Provider call audit records for completed and failed attempts.
- Stream error events are treated as failed attempts for retry/fallback audit.
- Streaming attempts are budget-checked before events are emitted by the center.

### Tools

- `ToolSpec`, `ToolRegistry`, and `ToolCenter`.
- Runtime mounts.
- Tool tags, aliases, manifests, inventory, and search.
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

### Skills

- `SkillRegistry` and `SkillsContext`.
- Markdown skill parsing.
- `SKILL.md` discovery.
- Zip skill archive loading with path-safety checks.
- Resource loading and windowed context views.

### MCP

- `MCPCenter`.
- Tool, resource, and prompt registration.
- Server state refresh.
- SDK-free stdio JSON-RPC connector.

### Memory

- `MemoryCenter` and `MemoryPort`.
- In-memory memory store.
- SQLite memory store.
- Markdown memory store.
- Postgres, vector DB, graph, or product memory can implement `MemoryPort` outside core.
- Runner-level memory recall injection through `ContextInjection`.
- Memory governance hooks.
- Leak scanning and global bucket checks.

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
PG, graph memory, multi-tenant isolation, retention policy, or product audit.

### Planner

- `PlannerPort` for plan-and-execute agents.
- `Plan`, `PlanStep`, and `PlanUpdate` state contracts.
- `InMemoryPlanner` for tests, examples, and lightweight embeddings.
- `PersistentPlanner` backed by `PlannerStorePort`.
- `InMemoryPlannerStore`, `SQLitePlannerStore`, and `MarkdownPlannerStore`.
- Dependency-aware ready-step selection.
- `PlanExecutor` for sequential ready-step execution through `AgentSessionManager`.
- `PlanExecutionReport` and `PlanExecutionStep` manifests for audit.
- Manifest export with terminal state, ready steps, and status counts.

The SDK owns generic plan state, update semantics, sequential step execution,
and single-process manager capacity guards. Runtimes own the actual planning
strategy: Raven can produce pentest plans, a code agent can produce repair
plans, and an ops agent can produce incident response plans while all of them
reuse the same core contract. Runtime code also owns distributed scheduling,
cross-worker concurrency, retry policy, and business workflow rules.

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

`PromptIR.trim_to_budget()` trims lower-priority dynamic buckets first and
records trim metadata in the prompt manifest. `AgentRunner` applies this against
`RuntimeBudget.max_prompt_bytes` before calling the provider.

If `AgentSession.context_reducer` is configured, `AgentRunner` first applies
timeline reduction against `RuntimeBudget.max_timeline_bytes`. This gives the SDK
Yaklang-style automatic context compaction without forcing a Raven-specific
summarizer or storage backend into core.

`ContextInjection` lets runtimes or core services place structured material into
a target bucket without rewriting the prompt builder. Resume checkpoints use this
path today. `AgentRunner` also injects memory recall through this path when
memory is enabled; operator hints and runtime-specific context can use the same
SDK-level mechanism later.

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

The runtime may be Raven, a code agent, an ops agent, or any other host. The runtime owns concrete tools, credentials, persistence, UI, and deployment. `raven_heart` owns the reusable agent mechanics.

## Current Status

| Area | Status |
| --- | --- |
| Harness/ReAct core | MVP implemented |
| Provider center | MVP implemented |
| Provider call audit | MVP implemented |
| Trace correlation | MVP implemented |
| Tool center | MVP implemented |
| Tool replay store | MVP implemented |
| Skill center | MVP implemented |
| MCP center | MVP implemented |
| SQLite/Markdown memory | MVP implemented |
| Planner core | MVP implemented |
| Approval core | MVP implemented |
| Prompt buckets/trimming | MVP implemented |
| Context reducer | runner-integrated MVP |
| Runtime adapter code | intentionally excluded |
| Full OpenAI Agents SDK replacement | in progress |

## License

MIT. See `LICENSE`.
