# raven_heart

`raven_heart` is a provider-neutral agent core for building long-running
Harness/ReAct agents. It was extracted from RavenStorm so the agent base can be
developed independently from Raven's current OpenAI Agents SDK runtime.

中文一句话：这是 Raven 的 AI agent 基座，不是 pentest runtime。它负责 agent 怎么思考、
怎么组织上下文、怎么调工具、怎么接记忆、怎么恢复运行；具体工具、Graphiti、FastAPI、
OpenAI Agents SDK 这些都应该通过 runtime adapter 接进来。

## Why This Exists

RavenStorm originally used the OpenAI Agents SDK as the main runtime. That is
convenient early on, but it makes several core behaviors hard to own:

- ReAct loop semantics and action parsing.
- Harness lifecycle, checkpoint, resume, and audit trail.
- Prompt buckets and context trimming.
- Tool registry, MCP registry, skill loading, and policy gates.
- Memory stores and memory governance.
- Runtime migration across security, code, ops, and research agents.

`raven_heart` moves those behaviors into our own base layer. Raven-specific
runtime pieces stay outside the core.

## Design Boundary

| Belongs in `agent_core` | Belongs in runtime / integration |
| --- | --- |
| Harness state machine | FastAPI routes, WebSocket events |
| ReAct executor | Concrete task orchestration UI |
| Prompt bucket IR | Domain-specific prompt contracts |
| Tool center and tool manifests | Real pentest tools and sandbox execution |
| Skill registry and context windows | Skill distribution and product UX |
| MCP center and stdio connector | MCP server deployment and credentials |
| Provider port and provider router | OpenAI, Anthropic, local model clients |
| Memory center, SQLite, Markdown memory | Graphiti, Document RAG, project stores |
| Policy interface and rule policy | Operator approval UX and org policy |

The rule is simple: `agent_core` owns agent mechanics. Runtime owns concrete
systems, credentials, persistence, UI, and domain behavior.

## Core Capabilities

### Harness

- Run and turn lifecycle.
- Terminal status checks.
- In-memory journal.
- Checkpoints and resume tokens.
- Run manifest and error recording.

### ReAct

- Provider-neutral ReAct loop.
- Structured action parsing.
- Built-in actions such as `finish`, `fail`, `call_tool`, `search_skill`,
  `load_skill`, and memory actions.
- Loop guard.
- Tool replay.
- Prompt timeline.
- Memory injection.
- Artifact compaction.

### LLM Providers

- `LLMProviderPort` protocol.
- Provider registry and default provider routing.
- Retry and fallback.
- Usage and failure accounting.
- Budget checks.

### Tools

- `ToolSpec`, `ToolRegistry`, and `ToolCenter`.
- Runtime mounts.
- Tool tags, aliases, manifests, inventory, and search.
- In-memory replay support.

### Skills

- `SkillRegistry` and `SkillsContext`.
- Markdown skill parsing.
- `SKILL.md` discovery.
- Zip archive skill loading with path safety checks.
- Resource loading and windowed context views.

### MCP

- `MCPCenter`.
- Tool, resource, and prompt registration.
- Server state refresh.
- Stdio JSON-RPC connector.
- No dependency on the MCP Python SDK inside core.

### Memory

- `MemoryCenter` and `MemoryPort`.
- In-memory memory store.
- SQLite memory store.
- Markdown memory store.
- Memory governance hooks.
- Leak scanning and global bucket checks.

### Prompt Buckets

The prompt builder uses bucketed context instead of one giant string:

| Bucket | Purpose |
| --- | --- |
| Static | Stable role, rules, and agent contract |
| Capability | Available tools, MCP tools, actions, and skills |
| Skill | Loaded skill bodies and resource windows |
| Memory | Recalled facts, continuity hints, and prior observations |
| Task | Current objective, constraints, and runtime context |
| Reactive | Recent tool results, failures, deltas, and loop-local state |

This mirrors the useful part of Yaklang-style agent context management without
copying Yaklang code.

## Yaklang Influence

This project studies Yaklang's agent ideas at the architecture level:

- Semantic context partitioning.
- Automatic context trimming.
- Tool and skill capability surfaces.
- Memory-backed continuity.
- Runtime adapter separation.

It does **not** translate Yaklang source code. Yaklang is AGPL-licensed; this
repository should remain an independent implementation of similar concepts.

## Repository Layout

```text
agent_core/
  actions.py          # action registry and validation
  harness.py          # lifecycle, journal, checkpoint, resume
  react.py            # ReAct executor
  providers.py        # LLM provider ports and router
  tools.py            # tool registry and tool center
  skills.py           # skill registry and context windows
  mcp.py              # MCP center
  mcp_stdio.py        # stdio JSON-RPC MCP-like connector
  memory.py           # memory ports and stores
  context.py          # prompt/context builder
  prompt.py           # prompt IR and buckets
  policy.py           # policy ports and rule policy
  runner.py           # high-level session runner

tests/
  test_agent_core_*.py

integrations/
  raven/
    agent_core_adapter.py   # Raven runtime adapter
    test_runtime_adapter.py # requires Raven runtime modules

docs/
  architecture.md
```

## Install

For local development:

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

Run the standalone core suite:

```bash
python -m pytest
```

Current extraction baseline:

```text
102 passed
```

The default suite intentionally excludes Raven-specific integration tests.
`integrations/raven/test_runtime_adapter.py` requires a RavenStorm checkout with
`app.openai_agents_runtime` available.

## Raven Integration

The Raven adapter demonstrates how to expose existing Raven runtime objects
through `agent_core` ports:

- SDK function tools become `ToolRuntimePort` mounts.
- Runtime completion callables become `LLMProviderPort` providers.
- Working memory or Graphiti-like backends become `MemoryPort` stores.
- TaskTree tools can be probed before session construction.

This is the intended migration path:

1. Keep Raven runtime behavior where it is.
2. Wrap Raven tools, memory, and providers with adapters.
3. Run Raven flows through `AgentRunner`.
4. Move OpenAI Agents SDK usage behind the adapter.
5. Remove direct SDK dependency from the core path once parity is proven.

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
| Raven adapter | Prototype implemented |
| Full Raven runtime replacement | Not complete |

This repository is usable as a base, but it is not yet a finished replacement
for RavenStorm's OpenAI Agents SDK runtime. Full replacement still needs real
runtime parity tests, provider integration, production persistence, and Raven
end-to-end migration.

## License

MIT. See `LICENSE`.

