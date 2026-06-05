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

## SDK Package Manifest

`agent_core_sdk_manifest()` is the package-level contract that a host runtime
can inspect before integration. It reports the package version, root public API,
capability matrix, storage backend interface roles, built-in backend kinds, and
external backend kinds that runtimes or adapter packages may implement.

The same manifest also publishes the runtime boundary: `runtime imports
agent_core`, the concerns owned by core, the concerns owned by the host runtime,
and forbidden runtime dependencies/packages such as OpenAI Agents SDK, Raven
runtime adapters, Graphiti, FastAPI, Redis, SQLAlchemy, and third-party MCP SDK
imports. This gives RavenStorm or a future code-agent runtime a simple
machine-readable preflight check before replacing OpenAI Agents SDK behavior.

`AgentCoreReadinessProfile` turns that package manifest into a concrete
acceptance gate. The default `agent_core-replacement-readiness` profile requires
the generic core contracts for harness lifecycle, ReAct, provider routing,
tool/skill/MCP orchestration, prompt/context semantics, memory governance,
storage manifests, policy/approval, trace/replay/eval, and coordination. It also
requires the public API names a host runtime would call, built-in and external
storage backend kinds, and the declarations that keep runtime adapters out of
the SDK package. `evaluate_agent_core_readiness()` returns an
`AgentCoreReadinessReport` with matched contracts and blocking issues. This is a
core-SDK readiness check; product wiring, RavenStorm adapter behavior,
credentials, UI streams, and production data stores still need runtime-level
acceptance tests outside this repository.

`AgentCoreAPIContract` is the package-root compatibility gate. It declares the
stable names a runtime may depend on for integration, separates the remaining
exports as MVP surface, and documents the version policy.
`evaluate_agent_core_api_stability()` checks a package manifest against that
stable set and blocks if a stable name is missing. This does not freeze every
implementation detail before 1.0; it freezes the SDK entrypoints that Raven,
code agents, and ops agents need for migration work.
`evaluate_agent_core_api_lifecycle()` checks the same package manifest against
the API lifecycle policy. It keeps stable, MVP, experimental, and deprecated
exports machine-readable, reports the package version and pre-1.0 status, and
blocks unclassified or overlapping API groups before a runtime pins the SDK.

`AgentCoreRuntimeBoundaryReport` is the executable boundary audit. It scans the
SDK package for forbidden runtime imports and forbidden adapter package
directories declared by the manifest. `evaluate_agent_core_runtime_boundary()`
therefore turns "runtime adapters stay out of core" from a documentation rule
into a package-level validation gate.

`AgentCoreAcceptanceHarness` is the next gate after manifest readiness. It runs a
deterministic pure-core scenario through `AgentRunner`: a provider requests a
tool call, `ToolCenter` routes it, memory recall injects context, the run writes
journal/event/tool-replay/trace records, and `DefaultTraceEvaluator` checks the
resulting trace. `AgentCoreAcceptanceReport` aggregates readiness, trace eval,
trace summary, run summary, and blocking issues. The harness proves that the SDK
base can execute the generic agent loop end to end, while RavenStorm-specific
routes, credentials, distributed workers, UI streaming, and production storage
remain separate adapter/runtime acceptance concerns.

`AgentCoreContextAcceptanceHarness` is the context-semantics gate. It runs a
deterministic pure-core scenario for context material routing, semantic
selection, target denial, context injection, per-injection trimming,
bucket-local prompt budgets, semantic prompt trimming, provider-aware prompt
budgeting, and trace eval. It proves the Yaklang-inspired context pipeline is a
portable SDK behavior; runtime-specific stores, rankers, embeddings, domain
prompts, and product policies remain outside core.
`AgentCoreContextWindowAcceptanceHarness` verifies the unified context window
audit contract. It checks that selected, injected, trimmed, excluded,
over-budget, and missing-source conditions become one prompt-safe report that
can be consumed by Raven, code-agent, ops-agent, or security-agent runtimes.

`AgentCoreApprovalAcceptanceHarness` is the policy/approval gate. It verifies
that a policy rule can block a tool, create a prompt-safe pending approval
record, accept an approved decision as `ApprovalResumeContext`, inject that
approval into a resumed prompt, execute the gated tool, and expose approval plus
policy records to trace eval. The SDK owns the contract, queue primitives,
resume material, and acceptance report; runtimes own operator identity, UI,
notifications, ticketing, workflow routing, and organization policy sources.

`AgentCoreOrchestrationAcceptanceHarness` is the capability-orchestration gate.
It verifies unified discovery across actions, local tools, loaded skills, MCP
tools, MCP resources, MCP prompts, and MCP servers; `ToolCenter` routing and
execution for local and MCP-backed tools; MCP inventory refresh; MCP
context-material export; capability prompt rendering; and ToolCenter/MCP/skill
trace summaries. The SDK owns the portable registry, center, discovery, and
trace contracts; runtimes own concrete tools, MCP process lifecycle, secrets,
network sessions, sandboxing, and product workflow queues.

`AgentCoreCoordinationAcceptanceHarness` is the generic multi-agent
coordination gate. It verifies dependency-aware plan execution, capability-based
handoff selection, managed agent sessions exposed through `AgentToolRuntime`,
prompt-safe artifact records for large coordination evidence, and trace/eval
contracts for planner, handoff, agent-as-tool, and artifacts. The SDK owns the
portable coordination semantics; runtimes own distributed scheduling, workflow
queues, tenant policy, and domain-specific planning strategy.

`AgentCoreEventAcceptanceHarness` is the streaming/event gate. It verifies
streaming ReAct runs emit `model_stream` events, event logs can be paged and
tailed with `EventStreamCursor`, managed background runs expose events before
and after completion, and trace eval can require event-log types, terminal
events, and monotonic event sequencing. The SDK owns event schemas, cursor
semantics, lightweight persistence, and trace contracts; runtimes own SSE,
WebSocket, metrics, dashboards, and production observability sinks.

`AgentCoreExternalBackendAcceptanceHarness` is the backend portability gate. It
verifies built-in SQLite and Markdown backend manifests, runtime-owned
postgres/vector/graph/product backend contracts, external memory and
context-material calls, storage preflight selection, context injection, selected
context material, and trace eval. The SDK owns ports, manifests, selection, and
preflight contracts; runtimes own concrete PG, vector DB, graph/Graphiti,
product API, migration, tenancy, and retention drivers.

`AgentCoreGuardrailAcceptanceHarness` is the input/policy/output guardrail gate.
It verifies preflight blocking, policy denial for dangerous tools without
execution, structured-output validation and repair, policy decision audit
records, failure summaries, and trace eval. The SDK owns the portable guardrail
contracts and default deterministic checks; runtimes own product safety policy,
operator escalation, tenant rules, and domain-specific eval suites.

`AgentCoreLifecycleAcceptanceHarness` is the scheduling/lifecycle gate. It
verifies single-process capacity queueing and dequeue, queued-run cancellation
before capacity claim, active-run cancellation, request timeouts, interrupt
metadata, schedule snapshots, event logs, and trace eval. The SDK owns local
manager semantics and prompt-safe reports; runtimes own distributed schedulers,
leases, worker process control, UI controls, and operational policy.

`AgentCoreProviderAcceptanceHarness` is the provider-compatibility gate. It runs
a deterministic matrix for text, multimodal, structured-output, native-tool,
OpenAI-compatible codec, the default provider-neutral transport codec,
prompt-safe request manifests, streamed tool-call and retryable-error decoding,
transport fallback, and streaming routes. It proves the SDK can choose
providers by capability and preserve portable call manifests without embedding
concrete HTTP clients, credentials, rate limits, or vendor-specific adapters in
core.

`AgentCoreProviderConformanceHarness` is the adapter/provider conformance gate.
It accepts any external `LLMProviderPort` implementation and checks text
completion, streaming, JSON mode, and native tool-call behavior using
provider-neutral requests and prompt-safe reports. The package-level default
uses a deterministic no-network provider; host runtimes can pass real providers
when API keys are available. The SDK owns the conformance contract, while HTTP
clients, credentials, deployment routing, and rate-limit policy remain outside
core.
The conformance report includes a `check_matrix` that separates required,
optional, skipped, completed, and failed checks, records whether an external
provider was supplied, and lists the core contracts plus runtime responsibilities
for adapter smoke tests.

`AgentCoreBudgetAcceptanceHarness` is the provider-budget gate. It verifies
`LLMProviderCenter` estimated-cost preflight, actual-cost enforcement,
request-level budget overrides, call-attempt limits, token limits, usage
accounting, and trace eval for provider cost. The SDK owns these portable
budget semantics and prompt-safe reports; host runtimes own vendor quota
integrations, tenant rate limits, billing systems, and operator policy.

`AgentCoreRedactionAcceptanceHarness` is the prompt-safe redaction gate. It
verifies `RedactionPolicy` recursively redacts sensitive keys, bearer/API-key/
password-like values, inline secret strings, and oversized strings while keeping
digest, byte-count, and path audit decisions. The SDK owns the portable
redaction contract and report shape; host runtimes own secret stores,
tenant-specific redaction policy, credential rotation, and incident workflows.

`TraceEvalSuiteRunner` is the generic regression-suite gate. It evaluates
multiple `TraceEvalCase` objects, where each case binds one run ID to its own
`TraceEvalSpec`, tags, and metadata, then returns one aggregate
`TraceEvalSuiteReport`. Missing traces are reported as blocked suite cases
instead of surfacing as host-runtime exceptions. The SDK owns the suite
execution and report shape; runtimes own domain datasets, dashboards, baseline
selection, and scoring policy.

`AgentCoreStorageAcceptanceHarness` is the storage-portability gate. It verifies
SDK built-in in-memory, SQLite, Markdown, and none backend manifests across
store roles; SQLite and Markdown memory/context roundtrips; backend extraction
from component manifests; runtime-owned Postgres, vector, object-storage, and
product backend contracts; positive multi-role storage preflight; and negative
blocked preflight behavior. The SDK owns backend metadata, selection, and
preflight contracts; runtimes own concrete PG/vector/graph/object-store/product
clients, migrations, tenancy, retention, and operational policy.

`AgentCoreRecoveryHarness` adds the recovery gate. It runs provider fallback and
tool retry scenarios with deterministic failures, then returns an
`AgentCoreRecoveryReport` containing retry counts, fallback use, attempt
statuses, and error-classification kinds. The SDK owns the portable recovery
audit shape and default acceptance checks; host runtimes own real retry budgets,
rate-limit handling, circuit breakers, vendor-specific failover, and operator
policy.

`AgentCoreDurableSessionAcceptanceHarness` adds the integrated durable-session
gate. It runs the same `AgentRunner` path through `AgentSessionManager` over
SQLite stores and Markdown stores, then reopens those stores and verifies
journal/checkpoint/trace, event log, tool replay, memory, context material,
policy decisions, and manager run state can still be queried. The SDK owns this
local durable contract and the external backend ports; runtimes own production
PG/vector/graph/object-store/product implementations and operations.

`AgentCoreResumeAcceptanceHarness` adds the checkpoint/resume gate. It creates a
checkpoint, builds a `ResumePlan`, resumes through `AgentRunner.resume()`,
checks prompt injection of the checkpoint state, runs trace eval for resume and
resume-plan readiness, and verifies strict terminal checkpoints are rejected
when `allow_terminal=False`. The SDK owns resume tokens, plans, prompt-safe
resume manifests, and the acceptance report; runtimes own distributed worker
selection, UI recovery flows, and product workflow state.

`AgentCoreNativeToolAcceptanceHarness` adds the provider-native tool-call gate.
It runs the real `AgentRunner` native-tool path against deterministic provider
responses, verifies provider tool contracts, executes the provider tool call
through the SDK tool runtime, returns a provider-neutral `role=tool` message,
preserves the originating `provider_tool_call` metadata in the journal, and
requires trace eval coverage for provider tool calls and provider tool results.
The SDK owns the provider-neutral loop contract; host runtimes own vendor HTTP
payloads, credentials, deployment routing, and concrete provider adapters.

`AgentCorePackagingAcceptanceHarness` adds the package-readiness gate. It checks
`pyproject.toml` project metadata, explicit build-system configuration, zero
runtime dependencies, package discovery for `agent_core`, stable public imports,
required repository files, and example entrypoint smoke structure. The SDK owns
these release-facing contracts; release automation, package indexes, signing,
and adapter wheels stay outside core.

`AgentCoreTaskProfileAcceptanceHarness` adds the cross-runtime task-profile
gate. It runs code, ops, and security task profiles through the same
`AgentRunner`, provider center, ToolCenter, SkillsContext, memory recall,
context-material selection, event log, and trace-eval pipeline. The SDK owns the
portable task mechanics; runtimes own concrete code tools, ops integrations,
security tooling, domain prompts, and product workflows.

`AgentCoreValidationSuite` is the aggregate package gate. It runs runtime
boundary audit, readiness, API stability, replacement acceptance, context
acceptance, approval acceptance, orchestration acceptance, coordination
acceptance, durable-session acceptance, event acceptance, external-backend
acceptance, guardrail acceptance, lifecycle acceptance, native-tool acceptance,
packaging acceptance, provider acceptance, provider-conformance checks,
budget acceptance, redaction acceptance, eval-suite checks, storage acceptance, task-profile
acceptance, recovery acceptance, and resume acceptance, then returns one
`AgentCoreValidationReport` with all subreports and blocking issues. This is the
SDK-level check a runtime should pass before adapter implementation or
product-specific migration tests begin.

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
| Portable state bundle contracts | Concrete migration jobs, cross-tenant copy policy, external storage drivers |
| Run trace bundle | Durable trace export, observability pipeline, retention |
| Trace replay/eval harness and generic eval suite runner | Domain datasets, dashboards, regression policy |
| Manager run state and `AgentRunStorePort` | Product workflow DB, scheduling, distributed workers |
| Multi-agent handoff protocol | Product queues, distributed workers, business orchestration |
| Prompt buckets, trimming, and injection | Domain-specific prompt material and task contracts |
| Prompt bucket budget policy | Domain-specific budget numbers and release gates |
| Prompt-safe redaction policy | Secret stores, tenant-specific policy, credential rotation |
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
| Context material | `ContextMaterialStorePort` | In-memory, SQLite, Markdown | PG, vector DB, graph/RAG, product APIs |
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
The SDK manifest also exposes `storage_backend_interfaces.role_contracts`, a
per-role map from store role to SDK port, core builtin kinds, runtime-owned
kinds, and the runtime responsibility note. This is the machine-readable line
between `agent_core` and host-owned PG/vector/graph/object-store/product
adapters.

`StorageBackendCatalog` is the SDK boundary for choosing among those backends.
It registers built-in and runtime-owned backend manifests, evaluates
`StorageBackendRequirement` constraints, and returns a
`StorageBackendSelection` with candidate scores and rejection reasons. A runtime
can therefore preflight "memory must be tenant-a, durable, semantic, vector
capable" without putting a PostgreSQL, graph, or vector client inside
`agent_core`.
For full startup checks, `StorageBackendCatalog.preflight()` returns a
`StorageBackendPreflightReport` covering multiple required roles. It records
selected roles/kinds, missing roles, blocking reasons, and every per-role
selection so Raven, code agents, or ops agents can prove their PG/vector/graph
or SQLite/Markdown choices are ready before a run starts.
`storage_backend_manifests_from_components()` extracts backend manifests from
component manifests such as memory centers, context material centers, session
manifests, or trace bundles. `storage_backend_catalog_from_components()` then
turns those into a catalog, so runtime-owned external stores can participate in
preflight through normal SDK manifests instead of adapter imports.

`AgentStateBundleBuilder` is the SDK state migration/archive boundary. It takes
generic component manifests for roles such as journal, run state, memory, event
log, run trace, policy, planner, tool replay, approval, or artifacts and returns
a prompt-safe `AgentStateBundle` with redaction audit, payload digests, byte
counts, required-role checks, and a role-ordered restore plan. Runtimes own the
actual copy jobs, storage credentials, tenant isolation, retention policy, and
PG/object-store/product adapters.

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
`TraceExportBuilder` and `export_trace_bundle` add the SDK-level export boundary:
run traces and eval manifests are converted into prompt-safe records with
redaction decisions, byte counts, and SHA-256 digests before a runtime writes
them to files, CI artifacts, dashboards, or product audit stores.

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
`RunTraceQuery` is the matching retrieval surface for trace stores. The
in-memory, SQLite, and Markdown stores support status, run-id, metadata, limit,
and reverse-order filters through one contract; production PG/object-storage
trace indexes can implement the same port without changing replay/eval code.

`TraceReplayHarness`, `TraceReplayComparator`, `TraceEvalHarness`, and
`TraceEvalSuiteRunner` turn run trace manifests into deterministic replay steps,
baseline diff reports, provider-neutral evaluation reports, and multi-case
regression reports. Replay steps include resume selection,
checkpoint loading, journal events, event-log entries, provider calls, provider
request-shape payloads, provider streaming calls using prompt-safe summaries,
prompt budget, prompt bucket budget, semantic prompt trim, global prompt trim,
preflight reports, approval records, artifact records, structured output
validation/repair records, handoff records, provider-native tool result
records, embedding calls, and lifecycle hook records. The core
checks generic contracts such as final status, iteration limits, provider call
limits, embedding call limits, required embedding providers/models/dimensions,
provider-native tool-call presence/names, provider request-shape plans and
output-token ceilings, provider route preflight readiness/candidates/reasons,
provider/tool error classification kinds, unified failure-summary sources/kinds,
provider-native tool-result
presence/status/execution,
required events, required tools,
cost ceilings, event ordering, journal integrity, preflight status/issue-code constraints,
resume-plan presence,
resume-plan readiness, expected resume checkpoint ids, handoff status/session constraints, tool execution presence,
tool retry, minimum tool attempt counts, ToolCenter selected mount/tool and
failed-call constraints, required storage backend roles/kinds,
forbidden backend kinds, external-backend limits, storage backend preflight
readiness/roles/blocking reasons, event-log presence/types,
terminal events, sequence monotonicity, duplicate sequence limits, lifecycle
hook event/status/failure constraints, required context injection
names/sources/targets/statuses, required included/trimmed/excluded injection
sources, forbidden injection sources/statuses,
trimmed/excluded injection limits, memory governance allow/rewrite/deny and risk
ceilings, prompt budget source/provider/provider-limited/byte ceilings, prompt
bucket budget roles/statuses/over-budget ceilings, runtime semantic prompt trim
roles/statuses/dropped-unit ceilings, approval
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
`AgentRunQuery` is the portable retrieval contract for manager-run state. The
built-in in-memory, SQLite, and Markdown stores support run-key, session,
status, metadata-equality, limit, and reverse-order filters. Runtime-owned PG,
workflow database, or distributed scheduler stores can implement the same port
without changing `AgentSessionManager` callers.
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
`AgentRunPreflightCenter` is the earlier SDK guardrail bus. It runs after an
optional capability refresh and before resume, memory recall, prompt shaping, or
provider calls. Built-in checks validate request-local requirements such as
task-byte limits, required tools/actions/skills/MCP servers, and memory
availability. Runtime-owned tenant, RBAC, quota, ticket, or deployment gates can
mount additional checks without changing `AgentRunner`.
Provider route preflight uses the same static bus when a provider exposes
`route_plan()`: the runner asks for an adapter-free `LLMProviderRoutePlan` using
the requested provider/model, streaming flag, and structured-output
requirements, then blocks before prompt shaping if no route is ready. This keeps
model capability gates in the SDK while credentials, rate limits, concrete HTTP
clients, and vendor fallback policy remain adapter/runtime concerns.
Storage backend requirements use the same bus: a run request can provide
`StorageBackendRequirement` entries, the runner builds a catalog from session
component manifests, records a `StorageBackendPreflightReport`, and blocks
before provider calls if required PG/vector/graph/SQLite/Markdown backends are
not ready. The check remains adapter-free because it only reads prompt-safe
manifests.

`HandoffSpec`, `HandoffRequest`, `HandoffRouter`, and `MultiAgentCoordinator`
provide a provider-neutral multi-agent handoff contract. The core can advertise
session capabilities, select targets by tags/tools/skills/priority, run the
selected session through `AgentSessionManager`, and record handoff manifests.
Runtime code owns product queues, cross-process scheduling, retries, UI
orchestration, and domain delegation strategy.
`AgentToolRuntime` is the agent-as-tool variant of the same boundary: it turns
managed sessions into ordinary `ToolSpec` entries and invokes them through
`AgentSessionManager`, so parent agents can delegate using the existing ReAct
tool loop. Distributed worker pools, process isolation, queueing, and tenant
routing still live outside core. `AgentToolTrace` and trace/eval constraints
summarize those calls by tool name, target session, child run id, status, task
bytes, output bytes, and failure count without importing runtime workers.

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
estimated token/output size. For otherwise compatible routes, it can also shape
the request before the adapter call, such as capping `max_output_tokens` to the
declared model output limit. Routed call manifests retain both the selected
capability profile and the `LLMRequestShapePlan`, and trace evals can require
specific provider names, model names, or capabilities such as
`structured_output`, `json_mode`, `tool_calls`, and `streaming`.
`LLMProviderRoutePlan` is the preflight form of the same decision. It lists each
candidate provider, selected route, fallback candidates, capability/model
rejection reasons, requested provider/model, and streamed mode without invoking
a model. Provider call records attach the route plan manifest so traces can
explain why a runtime chose or skipped a provider while concrete credentials,
tenant routing, and vendor clients stay outside the SDK.
`AgentPromptBudgetPlan` lets the runner use those route capabilities before the
model call. When a selected provider exposes a context window and output limit,
the core derives a smaller effective prompt byte budget and records the decision
in prompt and run trace manifests.
`LLMRequestShapePlan` is the provider-center counterpart to that runner prompt
budget: the runner can shrink the prompt, while the provider center constrains
the outgoing generation request. Concrete adapters still own HTTP payloads,
credentials, deployment names, tenant routing, and vendor-specific quota logic.
Replay payloads include this plan on provider call steps, and trace evals can
require adjusted plans, specific shape decisions, provider names, and final
output-token ceilings.
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
The journal keeps each provider-native tool result tied to the originating
`provider_tool_call` manifest plus any `tool_execution` summary. Trace replay
emits derived provider-tool-result steps, and evals can require result names,
statuses, execution summaries, or failure ceilings.
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
`AgentCoreProviderAcceptanceHarness` exercises both the default transport codec
and the OpenAI-compatible codec without network access. Its contract matrix
checks provider-neutral payload schemas, content-part preservation, prompt-safe
request manifests, decoded native tool calls, streamed tool-call chunks, and
retryable stream errors before a runtime brings real API keys or vendor clients.
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
Failed provider call records also include `ErrorClassification` manifests so
runtime UIs, evals, and policy code can reason about retryable provider failures
without storing raw vendor payloads.
Runner streaming is an SDK-level switch, not a runtime adapter:
`AgentSession.stream` sets the default and `AgentRunRequest.stream` can override
one run. The same ReAct loop, event log, journal model events, and trace
metadata are used whether a runtime later exposes those events through SSE,
WebSocket, polling, or logs.

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
Tool attempts and failed final results include `ErrorClassification` manifests
for schema failures, exceptions, and failed tool results.
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
stores and runtime-owned PG, vector, graph, or product API
stores through `ContextMaterialStoreSpec`, `ContextMaterialQuery`,
`ContextMaterialRoute`, and `ContextMaterialSearchPlan`. The SDK ships
in-memory, SQLite, and Markdown stores plus `ExternalContextMaterialStore` for
prompt-safe call audit; production PG/vector/graph/product drivers remain
outside core. `AgentRunRequest.context_material_query` lets `AgentRunner`
collect those candidates explicitly and pass them through the existing
selector/trimming/injection pipeline.

`ReActExecutor` emits monotonic `AgentEvent.sequence` values. Lightweight users
can capture them with `ListEventSink`; inspectable or local durable runs can use
`SQLiteEventSink` or `MarkdownEventSink`. Production runtimes should implement
`EventSinkPort` or `EventLogPort` for their own logs, UI streams, metrics,
retention policy, or audit systems. `EventStreamCursor` and `EventStreamBatch`
give those runtimes a provider-neutral paging contract with ReAct `run_id`,
manager `run_key`, session, sequence, and event-type filters, plus `has_more`
and terminal-run detection. `EventStreamTail` drains a bounded number of
available pages and returns the next cursor for polling loops.
`AgentSessionManager.event_batch()` and `event_tail()` expose those contracts
for one managed background run before completion by tagging emitted events with
`run_key` and `session_name`; the SDK does not own the SSE/WebSocket transport.

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

`AgentRunner` applies `PromptIR.trim_to_budget()` with the effective prompt
budget before provider calls, so automatic trimming is part of the core
execution path rather than a Raven-specific adapter behavior. The default
effective budget starts from `RuntimeBudget.max_prompt_bytes` and can be
tightened by provider route capabilities through `AgentPromptBudgetPlan`.
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
`TraceExportBundle` preserves those evaluable trace shapes after redaction, so
CI, runtime dashboards, or replay archives can consume exported records without
receiving raw provider headers, API keys, tool secrets, or oversized prompt
strings.

`ContextInjectionPolicy` is the SDK-level guardrail applied before prompt
assembly. It can restrict target buckets, trim each injected block, cap total
injection bytes, and emit per-injection decisions into the prompt manifest. This
keeps resume, memory, approval, and runtime-supplied context on one auditable
path without adding Raven-specific prompt content to core.

`ContextWindowBuilder` derives `ContextWindowReport` from a prompt manifest or a
run trace bundle. The report rolls up material selection, injection decisions,
bucket byte counts, semantic trim, bucket budget, and global prompt trim into a
single prompt-safe context window audit. `ContextWindowPolicy` can flag
over-budget prompts, excessive context exclusions/trims, or missing required
sources while keeping concrete ranking models and product-specific context
stores outside `agent_core`.

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
