# raven_heart

`raven_heart` is a pure, provider-neutral Harness/ReAct LLM Agent SDK base.
It is designed to be imported by different runtimes such as Raven, code agents,
ops agents, research agents, and future automation systems.

中文一句话：这是一个纯独立 agent SDK 基座，只负责 agent 的通用核心能力，不包含 Raven、OpenAI Agents SDK、Graphiti 或其他具体 runtime adapter。

## Scope

`agent_core` contains only reusable agent mechanics:

- ReAct loop and structured action execution.
- Harness lifecycle, checkpoint, resume, trace, and replay primitives.
- LLM provider abstractions and provider routing.
- Tool registry and tool center.
- Skill center and skill context injection.
- MCP center and SDK-free stdio connector.
- Prompt buckets and context trimming.
- SQLite, Markdown, and in-memory stores for lightweight memory.
- Policy gates, budget metadata, loop guards, and capability manifests.

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
| Provider protocol | Concrete LLM clients and credentials |
| Tool registry protocol | Real tools, sandboxing, permissions |
| Skill registry | Skill distribution UX |
| MCP center interfaces | MCP server deployment and secrets |
| SQLite/Markdown memory | Graphiti, RAG, durable product stores |
| Harness state contracts | API routes, UI events, persistence backend |

The dependency direction must always be:

```text
runtime imports agent_core
agent_core never imports runtime
```

## Core Capabilities

### Harness

- Run and turn lifecycle.
- In-memory journal.
- Checkpoints and resume tokens.
- `AgentRunRequest.resume_token` for injecting checkpoint state into the next run.
- Run manifest and error recording.
- Terminal status checks.

### ReAct

- Provider-neutral ReAct loop.
- Structured action parsing.
- Built-in actions: `finish`, `fail`, `call_tool`, `search_skill`, `load_skill`, memory actions.
- Loop guard.
- Tool replay.
- Prompt timeline.
- Memory injection.
- Artifact compaction.

### LLM Providers

- `LLMProviderPort` protocol.
- Provider registry and routing.
- Retry and fallback.
- Usage/failure accounting.
- Budget checks.

### Tools

- `ToolSpec`, `ToolRegistry`, and `ToolCenter`.
- Runtime mounts.
- Tool tags, aliases, manifests, inventory, and search.
- In-memory replay.

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
- Memory governance hooks.
- Leak scanning and global bucket checks.

### Prompt Buckets

| Bucket | Purpose |
| --- | --- |
| Static | Stable role, rules, and agent contract |
| Capability | Available tools, MCP tools, actions, and skills |
| Skill | Loaded skill bodies and resource windows |
| Memory | Recalled facts, continuity hints, prior observations |
| Task | Current objective, constraints, and runtime context |
| Reactive | Recent tool results, failures, deltas, loop-local state |

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

Current baseline:

```text
102 passed
```

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
| Tool center | MVP implemented |
| Skill center | MVP implemented |
| MCP center | MVP implemented |
| SQLite/Markdown memory | MVP implemented |
| Prompt buckets/trimming | MVP implemented |
| Runtime adapter code | intentionally excluded |
| Full OpenAI Agents SDK replacement | in progress |

## License

MIT. See `LICENSE`.
