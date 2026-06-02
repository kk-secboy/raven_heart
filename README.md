# raven_heart

Provider-neutral AI agent core extracted from RavenStorm.

The core package is intentionally independent from OpenAI Agents SDK, FastAPI,
Graphiti, Raven runtime tools, and concrete provider implementations. Those
belong in adapters or runtime integrations.

## What is inside

- Harness lifecycle, journal, checkpoint, resume, and run status primitives.
- ReAct execution loop with action parsing, policy checks, loop guards, replay,
  prompt timeline, memory injection, and artifact compaction.
- LLM provider center with routing, retry/fallback, usage accounting, and budget checks.
- Tool registry and tool center with mounts, tags, aliases, manifests, and replay.
- Skill registry and context windowing for Markdown/SKILL.md-style skills.
- MCP center and stdio JSON-RPC connector.
- Memory center with in-memory, SQLite, and Markdown-backed stores plus governance.
- Prompt bucket builder for static, routing, skill, memory, tool, and reactive context.

## Layout

```text
agent_core/          # standalone core package
tests/               # core tests
integrations/raven/  # RavenStorm runtime adapter
docs/                # architecture notes
examples/            # small runnable examples
```

## Test

```bash
python -m pytest
```

The default test suite only covers the standalone core. Raven-specific adapter
tests live under `integrations/raven` and require a RavenStorm checkout/runtime.

