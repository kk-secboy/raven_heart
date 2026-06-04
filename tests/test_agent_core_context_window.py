from __future__ import annotations

import pytest

from agent_core.context_window import (
    ContextWindowBuilder,
    ContextWindowPolicy,
    build_context_window_report,
)
from agent_core.context_window_acceptance import run_agent_core_context_window_acceptance


def _trace() -> dict[str, object]:
    return {
        "schema_version": "agent-core-run-trace-bundle/v1",
        "prompt": {
            "schema_version": "agent-core-prompt-ir/v1",
            "prompt_bytes": 900,
            "prompt_sha256": "prompt-hash",
            "buckets": [
                {"role": "semi_dynamic_1", "bytes": 140},
                {"role": "timeline_open", "bytes": 300},
            ],
            "metadata": {
                "bucket_budget": {
                    "schema_version": "agent-core-prompt-bucket-budget-result/v1",
                    "trimmed_count": 1,
                    "decisions": [{"role": "semi_dynamic_1", "status": "trimmed"}],
                },
                "semantic_trim": {
                    "schema_version": "agent-core-prompt-semantic-trim-result/v1",
                    "trimmed_count": 1,
                    "decisions": [{"role": "timeline_open", "status": "trimmed"}],
                },
            },
        },
        "context_material_selection": {
            "selections": [
                {
                    "name": "memory-hit",
                    "role": "memory",
                    "target": "semi_dynamic_1",
                    "status": "selected",
                    "selected": True,
                    "score": 3.0,
                    "rank": 1,
                    "bytes": 260,
                    "sha256": "memhash",
                    "metadata": {"source": "memory"},
                },
                {
                    "name": "runtime-hint",
                    "role": "runtime",
                    "target": "timeline_open",
                    "status": "selected",
                    "selected": True,
                    "score": 2.0,
                    "rank": 2,
                    "bytes": 80,
                    "sha256": "runtimehash",
                    "metadata": {"source": "runtime"},
                },
                {
                    "name": "denied-static",
                    "role": "system",
                    "target": "high_static",
                    "status": "target_denied",
                    "selected": False,
                    "rank": 0,
                    "bytes": 40,
                    "sha256": "deniedhash",
                    "metadata": {"source": "runtime"},
                },
            ]
        },
        "context_injections": {
            "injections": [
                {
                    "name": "memory-hit",
                    "source": "memory",
                    "target": "semi_dynamic_1",
                    "status": "trimmed",
                    "included": True,
                    "trimmed": True,
                    "original_bytes": 260,
                    "final_bytes": 120,
                },
                {
                    "name": "runtime-hint",
                    "source": "runtime",
                    "target": "timeline_open",
                    "status": "included",
                    "included": True,
                    "trimmed": False,
                    "original_bytes": 80,
                    "final_bytes": 80,
                },
            ]
        },
    }


def test_context_window_report_rolls_up_selection_injection_and_trim() -> None:
    report = ContextWindowBuilder(
        policy=ContextWindowPolicy(
            max_prompt_bytes=1000,
            max_excluded_contexts=1,
            max_trimmed_contexts=1,
            required_sources=("memory", "runtime"),
        )
    ).from_trace(_trace())
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-context-window-report/v1"
    assert manifest["ready"] is True
    assert manifest["prompt_bytes"] == 900
    assert manifest["bucket_bytes"] == {"semi_dynamic_1": 140, "timeline_open": 300}
    assert manifest["selected_count"] == 2
    assert manifest["included_count"] == 2
    assert manifest["trimmed_count"] == 1
    assert manifest["excluded_count"] == 1
    assert manifest["sources"] == {"memory": 1, "runtime": 2}
    assert manifest["statuses"] == {"included": 1, "target_denied": 1, "trimmed": 1}
    assert manifest["entries"][0]["name"] == "memory-hit"
    assert manifest["entries"][0]["stages"] == ["injected", "selected", "trimmed"]
    assert manifest["prompt_bucket_budget"]["trimmed_count"] == 1
    assert manifest["prompt_semantic_trim"]["trimmed_count"] == 1


def test_context_window_report_blocks_strict_policy() -> None:
    report = build_context_window_report(
        _trace(),
        policy=ContextWindowPolicy(
            max_prompt_bytes=100,
            max_excluded_contexts=0,
            max_trimmed_contexts=0,
            required_sources=("memory", "mcp"),
        ),
    ).manifest()
    issue_codes = {issue["code"] for issue in report["issues"]}

    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert {
        "prompt_bytes_exceeded",
        "excluded_context_count_exceeded",
        "trimmed_context_count_exceeded",
        "required_context_source_missing",
    } <= issue_codes


@pytest.mark.asyncio
async def test_agent_core_context_window_acceptance_gate_passes() -> None:
    report = await run_agent_core_context_window_acceptance(
        metadata={"test": "context_window"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-context-window-acceptance-report/v1"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["ready_report"]["ready"] is True
    assert manifest["ready_report"]["selected_count"] == 3
    assert manifest["ready_report"]["trimmed_count"] == 2
    assert manifest["blocked_report"]["status"] == "blocked"
