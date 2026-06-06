# Yaklang-Style Loop Alignment Audit

This audit records the current RavenHeart evidence for the Yaklang-style loop
goal: fresh turn prompts, timeline/memory/perception feedback, dynamic context
selection, tools/skills/MCP, long-running work, topic switching, and knowledge
recall from embedded documents.

## Scope

This work stays inside the RavenHeart SDK. It does not add a RavenStorm adapter
or a Yaklang compatibility layer. The implementation reuses the existing
`AgentRunner`, `ReActExecutor`, `AgentPromptBuilder`, `TimelineStore`,
`MemoryPort`, `CapabilityCatalog`, tools, skills, and MCP center.

## Current Implementation Evidence

| Requirement | Evidence |
| --- | --- |
| Turn lifecycle owns fresh prompt generation | `agent_core/runner.py` has `_RunnerTurnContextRefresher.before_model_call(...)`, which performs memory recall, context material selection, timeline reduction prompt view, capability refresh, knowledge recall, midterm timeline recall, perception injections, and semantic trim before each model call. |
| Runner boundary is narrower | `AgentRunner.run(...)` wires session/executor/preflight/resume/trace and delegates per-turn prompt refresh to the runner refresher used by `ReActExecutor`. |
| Provider messages do not keep old full prompts | `agent_core/react.py` builds provider messages from current prompt segments plus compact loop delta. `examples/mock_yaklang_soak.py` and live reports measure compact delta bytes separately. |
| Timeline raw store is preserved while prompt view is reduced | `agent_core/timeline.py` exposes `TimelineStore`, `TimelineCursor`, `TimelineDiff`, and prompt-facing `TimelineView`; reduction updates prompt view/digest semantics instead of deleting original facts. |
| Perception happens after tool/action results | `agent_core/turn_runtime.py` implements `YaklangStylePerceptionController` and `ProviderBackedPerceptionEvaluator`; reports show perception provider calls and timeline perception entries. |
| Perception affects downstream recall | Runner refresher consumes perception downstream refresh to schedule memory, context material, knowledge, midterm timeline, and capability refresh. |
| Memory recall is resilient to slow embedding | `agent_core/runner.py` falls back from 200 ms semantic recall timeout to keyword recall and records `quick_semantic_timeout` metadata. |
| Native provider tool calls are compatible | `agent_core/providers.py` preserves provider tool-call payloads; `agent_core/react.py` executes native tool calls, maps core actions, repairs empty native finish calls, and supports direct text final answers. |
| Tools, skills, and MCP are visible and usable | Deterministic and live reports verify tool execution, skill context visibility, MCP server/resource context, and provider tool messages. |
| File-backed knowledge can be embedded and injected | `examples/knowledge_docs/*.md` are read by `examples/live_yaklang_e2e.py` into `InMemoryContextMaterialStore` with Embedding-3 semantic ranking. Live report records document path, bytes, and SHA-256. |
| Long-running work is bounded | `examples/mock_yaklang_soak.py` runs 12 tasks and 23 provider requests; report records tail request byte range and compact delta bounds. |

## Deterministic Verification

These checks were executed with the bundled Python runtime because `uv` and
`pytest` are not available in the current shell:

```text
react native finish repair: ok
memory fallback recall: ok
yaklang lifecycle topic switch: ok
yaklang long loop stability: ok
focused tests ok
```

The focused checks directly execute:

| Test | Coverage |
| --- | --- |
| `tests/test_agent_core_react.py::test_react_executor_repairs_native_finish_without_output` | Empty native `finish {terminal:true}` is not accepted as completion; repair requires non-empty output. |
| `tests/test_agent_core_runner.py::test_agent_runner_memory_recall_falls_back_to_keyword_after_quick_semantic_timeout` | Slow semantic memory recall falls back to keyword recall and still injects `[memory]`. |
| `tests/test_agent_core_yaklang_lifecycle.py::test_yaklang_style_functional_topic_switch_recall_and_prompt_blocks` | A -> B -> A functional topic switch, tools, timeline, memory, knowledge, MCP, skills, and prompt segments. |
| `tests/test_agent_core_yaklang_lifecycle.py::test_yaklang_style_long_loop_keeps_prompt_blocks_stable_and_recalls_old_topic` | Multi-task long loop with bounded provider messages, stable high-static/capability prefix, dynamic timeline views, and old-topic memory recall. |

## Live Verification

Live report:

```text
E:\ravenstrom\local-artifacts\live-yaklang-e2e-v12.json
```

Verifier:

```text
examples/verify_yaklang_reports.py E:\ravenstrom\local-artifacts\live-yaklang-e2e-v12.json
```

The live scenario uses:

| Component | Details |
| --- | --- |
| Chat model | DeepSeek `deepseek-v4-pro` |
| Embedding model | Zhipu `embedding-3`, 512 dimensions |
| Tools | `inspect_alpha`, `inspect_beta`, `exec_echo` |
| Skills | `alpha-callback-review`, `beta-billing-review` |
| MCP | In-memory server with tools, resources, and prompts |
| Knowledge docs | `examples/knowledge_docs/alpha_callback_state.md`, `beta_billing_webhook.md`, `unrelated_cache.md` |

Key live assertions are machine-checked in the report:

| Assertion | v12 status |
| --- | --- |
| all runs completed | true |
| alpha/beta tool facts in timeline | true |
| alpha/beta tool facts in memory | true |
| alpha return prompt contains memory | true |
| alpha return prompt contains alpha fact | true |
| tool result visible in provider request | true |
| skills visible in prompt | true |
| MCP visible in prompt | true |
| knowledge or midterm timeline used after topic switch | true |
| Embedding-3 used | true |
| perception used | true |
| high static prompt segment stable | true |
| frozen capability prefix stable | true |
| semi-dynamic and timeline-open segments change | true |
| max compact delta bytes | 276 |

## Long-Run Soak Verification

Deterministic soak report:

```text
E:\ravenstrom\local-artifacts\mock-yaklang-soak.json
```

Verifier:

```text
examples/verify_yaklang_reports.py E:\ravenstrom\local-artifacts\mock-yaklang-soak.json
```

The soak runs 12 tasks, 23 provider requests, and 11 tool calls. It records:

| Metric | Value |
| --- | --- |
| timeline item count | 94 |
| memory record count | 12 |
| max request bytes | 16381 |
| tail request byte range | 215 |
| max compact delta bytes | 195 |
| request growth bounded | true |
| compact delta bounded | true |
| old alpha topic recalled from memory | true |

## Machine-Checkable Gate

Both current reports pass:

```text
examples/verify_yaklang_reports.py \
  E:\ravenstrom\local-artifacts\mock-yaklang-soak.json \
  E:\ravenstrom\local-artifacts\live-yaklang-e2e-v12.json

ok: true
issues: []
```

The verifier checks the live and mock schemas independently. It fails if core
functional assertions, prompt stability assertions, knowledge document manifests,
bounded request growth, or compact delta limits regress.

## Remaining Gaps

The goal is not marked complete yet for these reasons:

- Full `pytest` has not been run in this environment. `uv` is unavailable and
  the bundled Python runtime does not include `pytest`.
- The live scenario is intentionally small to limit API cost. The longer soak is
  deterministic and no-network, not a live LLM soak.
- The audit proves current Heart behavior against the requested Yaklang-style
  requirements, but it does not claim byte-for-byte Yaklang parity or a Yaklang
  adapter.

